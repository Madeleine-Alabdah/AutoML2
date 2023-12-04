import logging
import os
import warnings

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

import dtreeviz
from sklearn.tree import _tree

from supervised.utils.subsample import subsample

from autokeras import StructuredDataClassifier, StructuredDataRegressor

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules

def save_rules(tree, feature_names, class_names, model_file_path, learner_name):
    try:
        rules = get_rules(tree, feature_names, class_names)
        fname = os.path.join(model_file_path, f"{learner_name}_rules.txt")
        with open(fname, "w") as fout:
            for r in rules:
                fout.write(r + "\n\n")
    except Exception as e:
        logger.info(f"Problem with extracting decision tree rules. {str(e)}")

class DecisionTreeAlgorithm(SklearnAlgorithm, BaseEstimator, ClassifierMixin):
    algorithm_name = "Decision Tree"
    algorithm_short_name = "Decision Tree"

    def __init__(self, params):
        super(DecisionTreeAlgorithm, self).__init__(params)
        logger.debug("DecisionTreeAlgorithm.__init__")
        self.library_version = sklearn.__version__
        self.max_iters = params.get("max_steps", 1)
        self.model = DecisionTreeClassifier(
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth", 3),
            random_state=params.get("seed", 1),
        )
        self.autokeras = params.get("autokeras", False)

    def file_extension(self):
        return "decision_tree"

    def interpret(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        model_file_path,
        learner_name,
        target_name=None,
        class_names=None,
        metric_name=None,
        ml_task=None,
        explain_level=2,
    ):
        super(DecisionTreeAlgorithm, self).interpret(
            X_train,
            y_train,
            X_validation,
            y_validation,
            model_file_path,
            learner_name,
            target_name,
            class_names,
            metric_name,
            ml_task,
            explain_level,
        )
        if explain_level == 0:
            return
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            try:
                if self.autokeras:
                    autokeras_model = StructuredDataClassifier(max_trials=10, epochs=100)
                    autokeras_model.fit(X_train, y_train)
                    self.model = autokeras_model
                else:
                    if len(class_names) > 10:
                        return

                    viz = dtreeviz.model(
                        self.model,
                        X_train,
                        y_train,
                        target_name="target",
                        feature_names=X_train.columns,
                        class_names=class_names,
                    )
                    tree_file_plot = os.path.join(
                        model_file_path, learner_name + "_tree.svg"
                    )
                    viz.view().save(tree_file_plot)
            except Exception as e:
                logger.info(f"Problem when visualizing decision tree. {str(e)}")

            save_rules(
                self.model, X_train.columns, class_names, model_file_path, learner_name
            )

dt_params = {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4]}

classification_default_params = {"criterion": "gini", "max_depth": 3}

additional = {
    "max_steps": 1,
    "max_rows_limit": None,
    "max_cols_limit": None,
    "autokeras": False,
}
required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    DecisionTreeAlgorithm,
    dt_params,
    required_preprocessing,
    additional,
    classification_default_params,
)

# Register AutoKeras algorithms
class AutoKerasClassifierAlgorithm(BaseAlgorithm, ClassifierMixin):
    algorithm_name = "AutoKeras Classifier"
    algorithm_short_name = "AutoKeras"

    def __init__(self, params):
        super(AutoKerasClassifierAlgorithm, self).__init__(params)
        self.max_iters = additional.get("max_steps", 1)
        self.model = StructuredDataClassifier(max_trials=10, epochs=100)

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        X_validation=None,
        y_validation=None,
        sample_weight_validation=None,
        log_to_file=None,
        max_time=None,
    ):
        self.model.fit(X, y, epochs=100)

    def is_fitted(self):
        return hasattr(self, "model") and self.model is not None

    def predict(self, X):
        self.reload()
        return self.model.predict(X)

# Register AutoKeras algorithms
AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    AutoKerasClassifierAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)
