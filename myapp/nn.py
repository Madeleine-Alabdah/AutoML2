#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import warnings
import numpy as np
import pandas as pd
from autokeras import StructuredDataClassifier, StructuredDataRegressor
from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
    AlgorithmsRegistry,
)
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class NNFit(BaseAlgorithm):
    def file_extension(self):
        return "neural_network"

    def is_fitted(self):
        return hasattr(self.model, "is_trained") and self.model.is_trained

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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore")
            self.model.fit(X, y, epochs=self.params.get("epochs", 100))

        if log_to_file is not None:
            # You can add logging for AutoKeras as needed
            pass


class MLPAlgorithm(NNFit):
    algorithm_name = "Neural Network"
    algorithm_short_name = "Neural Network"

    def init(self, params):
        super(MLPAlgorithm, self).init(params)
        logger.debug("MLPAlgorithm.init")
        self.model = StructuredDataClassifier(
            max_trials=params.get("max_trials", 10),
            objective=params.get("objective", "val_loss"),
        )

    def get_metric_name(self):
        return "logloss"


class MLPRegressorAlgorithm(NNFit):
    algorithm_name = "Neural Network"
    algorithm_short_name = "Neural Network"

    def init(self, params):
        super(MLPRegressorAlgorithm, self).init(params)
        logger.debug("MLPRegressorAlgorithm.init")
        self.model = StructuredDataRegressor(
            max_trials=params.get("max_trials", 10),
            objective=params.get("objective", "val_loss"),
        )

    def get_metric_name(self):
        return "mse"


nn_params = {
    "max_trials": [10, 20, 30],  # Adjust the search space as needed
    "objective": ["val_loss", "val_accuracy"],  # Adjust the search space as needed
}

default_nn_params = {"max_trials": 10, "objective": "val_loss"}

additional = {"max_rows_limit": None, "max_cols_limit": None}

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "scale",
    "target_as_integer",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    MLPAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    MLPAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "datetime_transform",
    "text_transform",
    "scale",
    "target_scale",
]

AlgorithmsRegistry.add(
    REGRESSION,
    MLPRegressorAlgorithm,
    nn_params,
    required_preprocessing,
    additional,
    default_nn_params,
)


# In[ ]: