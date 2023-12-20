import copy
import logging
import os
import time
from inspect import signature
import numpy as np
import pandas as pd
import xgboost as xgb
import autokeras as ak
from sklearn.base import ClassifierMixin, RegressorMixin

from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.utils.config import LOG_LEVEL
from supervised.utils.metric import (
    xgboost_eval_metric_accuracy,
    xgboost_eval_metric_average_precision,
    xgboost_eval_metric_f1,
    xgboost_eval_metric_mse,
    xgboost_eval_metric_pearson,
    xgboost_eval_metric_r2,
    xgboost_eval_metric_spearman,
    xgboost_eval_metric_user_defined,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

import tempfile

class XgbAlgorithmException(Exception):
    def __init__(self, message):
        super(XgbAlgorithmException, self).__init__(message)
        logger.error(message)

def time_constraint(env):
    pass

def xgboost_eval_metric(ml_task, automl_eval_metric):
    eval_metric_name = automl_eval_metric
    if ml_task == MULTICLASS_CLASSIFICATION:
        if automl_eval_metric == "logloss":
            eval_metric_name = "mlogloss"
    return eval_metric_name

def xgboost_objective(ml_task, automl_eval_metric):
    objective = "reg:squarederror"
    if ml_task == BINARY_CLASSIFICATION:
        objective = "binary:logistic"
    elif ml_task == MULTICLASS_CLASSIFICATION:
        objective = "multi:softprob"
    else:
        objective = "reg:squarederror"
    return objective

class XgbAlgorithm(BaseAlgorithm):
    algorithm_name = "Extreme Gradient Boosting"
    algorithm_short_name = "Xgboost"

    def __init__(self, params):
        super(XgbAlgorithm, self).__init__(params)
        self.library_version = xgb.__version__

        self.explain_level = params.get("explain_level", 0)
        self.boosting_rounds = additional.get("max_rounds", 10000)
        self.max_iters = 1
        self.early_stopping_rounds = additional.get("early_stopping_rounds", 50)
        self.learner_params = {
            "tree_method": "hist",
            "booster": "gbtree",
            "objective": self.params.get("objective"),
            "eval_metric": self.params.get("eval_metric"),
            "eta": self.params.get("eta", 0.01),
            "max_depth": self.params.get("max_depth", 1),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "n_jobs": self.params.get("n_jobs", -1),
            "seed": self.params.get("seed", 1),
            "verbosity": 0,
        }

        if "lambda" in self.params:
            self.learner_params["lambda"] = self.params["lambda"]
        if "alpha" in self.params:
            self.learner_params["alpha"] = self.params["alpha"]

        if self.learner_params["seed"] > 2147483647:
            self.learner_params["seed"] = self.learner_params["seed"] % 2147483647
        if "num_class" in self.params:
            self.learner_params["num_class"] = self.params.get("num_class")

        if "max_rounds" in self.params:
            self.boosting_rounds = self.params["max_rounds"]

        self.custom_eval_metric = None
        if self.params.get("eval_metric", "") == "r2":
            self.custom_eval_metric = xgboost_eval_metric_r2
        elif self.params.get("eval_metric", "") == "spearman":
            self.custom_eval_metric = xgboost_eval_metric_spearman
        elif self.params.get("eval_metric", "") == "pearson":
            self.custom_eval_metric = xgboost_eval_metric_pearson
        elif self.params.get("eval_metric", "") == "f1":
            self.custom_eval_metric = xgboost_eval_metric_f1
        elif self.params.get("eval_metric", "") == "average_precision":
            self.custom_eval_metric = xgboost_eval_metric_average_precision
        elif self.params.get("eval_metric", "") == "accuracy":
            self.custom_eval_metric = xgboost_eval_metric_accuracy
        elif self.params.get("eval_metric", "") == "mse":
            self.custom_eval_metric = xgboost_eval_metric_mse
        elif self.params.get("eval_metric", "") == "user_defined_metric":
            self.custom_eval_metric = xgboost_eval_metric_user_defined

        logger.debug("XgbLearner __init__")

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
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        X = X.drop(columns=to_drop)
        feature_selector = xgb.train(
            self.learner_params,
            xgb.DMatrix(
                X.values if isinstance(X, pd.DataFrame) else X,
                label=y,
                missing=np.NaN,
                weight=sample_weight,
            ),
            self.boosting_rounds,
        )
        selected_features = feature_selector.get_fscore()
        selected_features = sorted(selected_features, key=selected_features.get, reverse=True)
        selected_features = selected_features[:k]  # Set k to the desired number of selected features

        X_selected = X[selected_features]

        auto_model = ak.StructuredDataRegressor() 
        auto_model.fit(X_selected, y)

        self.auto_model = auto_model

    def predict(self, X):
        self.reload()

        if self.model is None:
            raise XgbAlgorithmException("Xgboost model is None")

        X_selected = X[selected_features]
        predictions = self.auto_model.predict(X_selected)

        return predictions

    def copy(self):
        return copy.deepcopy(self)

    def save(self, model_file_path):
        self.model.save_model(model_file_path)
        self.model_file_path = model_file_path
        logger.debug("XgbAlgorithm save model to %s" % model_file_path)

    def load(self, model_file_path):
        logger.debug("XgbLearner load model from %s" % model_file_path)
        self.model = xgb.Booster()
        self.model.load_model(model_file_path)
        self.model_file_path = model_file_path

    def file_extension(self):
        return "xgboost.json"

    def get_metric_name(self):
        metric = self.params.get("eval_metric")
        if metric is None:
            return None
        if metric == "mlogloss":
            return "logloss"
        return metric

class XgbClassifier(XgbAlgorithm, ClassifierMixin):
    pass

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    XgbClassifier,
    xgb_bin_class_params,
    required_preprocessing,
    additional,
    classification_bin_default_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    XgbClassifier,
    xgb_multi_class_params,
    required_preprocessing,
    additional,
    classification_multi_default_params,
)

regression_required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "target_scale",
]

class XgbRegressor(XgbAlgorithm, RegressorMixin):
    pass

AlgorithmsRegistry.add(
    REGRESSION,
    XgbRegressor,
    xgb_regression_params,
    regression_required_preprocessing,
    additional,
    regression_default_params,
)
