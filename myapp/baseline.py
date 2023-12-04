import logging
import os

import sklearn
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.algorithms.sklearn import SklearnAlgorithm
from supervised.utils.config import LOG_LEVEL

# Add AutoKeras import
from autokeras import StructuredDataClassifier, StructuredDataRegressor

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class BaselineClassifierAlgorithm(SklearnAlgorithm, ClassifierMixin):
    algorithm_name = "Baseline Classifier"
    algorithm_short_name = "Baseline"

    def __init__(self, params):
        super(BaselineClassifierAlgorithm, self).__init__(params)
        logger.debug("BaselineClassifierAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.max_iters = additional.get("max_steps", 1)
        self.model = DummyClassifier(
            strategy="prior", random_state=params.get("seed", 1)
        )


class BaselineRegressorAlgorithm(SklearnAlgorithm, RegressorMixin):
    algorithm_name = "Baseline Regressor"
    algorithm_short_name = "Baseline"

    def __init__(self, params):
        super(BaselineRegressorAlgorithm, self).__init__(params)
        logger.debug("BaselineRegressorAlgorithm.__init__")

        self.library_version = sklearn.__version__
        self.max_iters = additional.get("max_steps", 1)
        self.model = DummyRegressor(strategy="mean")


class AutoKerasClassifierAlgorithm(BaseAlgorithm, ClassifierMixin):
    algorithm_name = "AutoKeras"
    algorithm_short_name = "AutoKeras"

    def __init__(self, params):
        super(AutoKerasClassifierAlgorithm, self).__init__(params)
        self.max_iters = 1
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


class AutoKerasRegressorAlgorithm(BaseAlgorithm, RegressorMixin):
    algorithm_name = "AutoKeras"
    algorithm_short_name = "AutoKeras"

    def __init__(self, params):
        super(AutoKerasRegressorAlgorithm, self).__init__(params)
        self.max_iters = 1
        self.model = StructuredDataRegressor(max_trials=10, epochs=100)

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


additional = {"max_steps": 1, "max_rows_limit": None, "max_cols_limit": None}
required_preprocessing = ["target_as_integer"]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    BaselineClassifierAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    BaselineClassifierAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)

AlgorithmsRegistry.add(
    REGRESSION,
    BaselineRegressorAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)

# Register AutoKeras algorithms
AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    AutoKerasClassifierAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)
AlgorithmsRegistry.add(
    REGRESSION,
    AutoKerasRegressorAlgorithm,
    {},
    required_preprocessing,
    additional,
    {},
)
