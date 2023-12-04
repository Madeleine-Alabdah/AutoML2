# AutoKeras Algorithms

1.	Neural Network:
In the uploaded version of nn.py code in git , the neural network implementation has these new modifications:
-	The scikit-learn code utilized MLPClassifier and MLPRegressor for classification and regression tasks, respectively. On the other hand, the AutoKeras code incorporates StructuredDataClassifier and StructuredDataRegressor for similar tasks.
-	The edition introduces a more automated approach, where the model's architecture is determined through a search process. The number of trials and an optimization objective guide the model selection process.
-	The focus shifts to fewer hyperparameters, emphasizing the maximum number of trials and the optimization objective, streamlining the configuration process.
-	While logging is still supported, it is customizable based on the specific needs of AutoKeras.
-	The edition offers a more automated model selection process, potentially saving time and effort in exploring diverse model architectures.
-	This shift provides a dynamic perspective, allowing the selection of the most suitable framework based on the specific demands of the machine learning project.
2.	Random Forest

-	Two classes are defined: AutoKerasClassifier for classification and AutoKerasRegressor for regression.
These classes inherit from BaseAlgorithm and ClassifierMixin or RegressorMixin.
-	The AutoKeras models (StructuredDataClassifier and StructuredDataRegressor) are used with specified parameters like max_trials and directory for saving models.
-	AutoKeras algorithms are registered in the AlgorithmsRegistry for binary classification, multiclass classification, and regression tasks.
Hyperparameter search spaces are specified for the number of trials.

3.	Xgboost 

-	The fit method now utilizes AutoKeras for neural architecture search (NAS).
-	A small portion of the data is split for training an AutoKeras model (StructuredDataClassifier) using a limited number of trials (max_trials=3) and epochs (epochs=10).
-	The exported AutoKeras model is used to extract features or predictions from both the training and validation sets. The extracted features are converted into DataFrames and concatenated with the original features.
-	The combined dataset (original + AutoKeras features) is used to train the XGBoost model using the xgb.train method.
-	Training parameters are set based on the initial XGBoost configuration.
-	The dataset is used to create a DMatrix, and predictions are made using the trained XGBoost model.
-	This code introduces two new classes, AutoKerasClassifier and AutoKerasRegressor, which use the autokeras library for automated machine learning. These classes are designed to work with AutoKeras classifiers and regressors.

-	The first script includes parameters like max_trials for configuring the number of trials AutoKeras will perform during the search for the best model architecture.
4.	Linear

-	Structured Data Classifier and Regressor: we used AutoKeras's StructuredDataClassifier and StructuredDataRegressor for classification and regression tasks, respectively. These models are designed for structured data, making them suitable for tabular datasets.
-	We specified the max_trials parameter, allowing AutoKeras to explore different hyperparameter configurations during the model search. Users can adjust the number of trials based on their preferences or computational resources.
-	AutoKeras models are saved in the specified directory (autokeras_classifier and autokeras_regressor), allowing easy retrieval and reuse of the trained models.
-	We registered the AutoKeras models in the AlgorithmsRegistry for binary classification, multiclass classification, and regression tasks. This ensures consistency in how different algorithms are handled within your broader framework.
5.	Decision Tree
-	The code includes a logger for handling logs, and it uses the configuration settings from supervised.utils.config.
-	We added additional imports specific to AutoKeras, such as import dtreeviz. This necessary for AutoKeras-specific functionality or visualization.

-	The get_rules and save_rules functions are retained from the scikit-learn code. These functions are used for extracting decision tree rules and saving them to a text file. It's a valuable addition for interpretability.
-	The DecisionTreeAlgorithm and DecisionTreeRegressorAlgorithm classes are retained, but they are extended to work within the AutoKeras framework. They inherit from SklearnAlgorithm and the appropriate mixin classes.
-	The interpret method in both DecisionTreeAlgorithm and DecisionTreeRegressorAlgorithm now supports AutoKeras-specific functionality. It includes visualization of the decision tree using dtreeviz and saving the rules.
-	Default parameters for classification and regression are specified in classification_default_params and regression_default_params. The algorithms are registered with AutoKeras using the AlgorithmsRegistry class.
-	required_preprocessing and regression_required_preprocessing lists specify the preprocessing steps needed for classification and regression, respectively.
6.	Baseline:

-	A logger is included for handling logs, and it uses configuration settings from supervised.utils.config.
-	The BaselineClassifierAlgorithm and BaselineRegressorAlgorithm classes are adapted to work within the AutoKeras framework. They inherit from SklearnAlgorithm and the appropriate mixin classes.
-	Default parameters for the Baseline algorithm within AutoKeras are specified directly in the AutoKeras code..
-	The algorithms are registered with AutoKeras using the AlgorithmsRegistry class, ensuring they can be utilized seamlessly within AutoKeras.

-	The required_preprocessing list specifies the preprocessing steps needed for both binary and multiclass classification, ensuring proper data preprocessing compatibility with AutoKeras.

7.	LightGBM
 
-	The code includes a logger for handling logs, and it uses configuration settings from supervised.utils.config.
-	The LightgbmAlgorithm class is adapted to work within the AutoKeras framework. It inherits from BaseAlgorithm and includes adjustments to parameters and configurations suitable for AutoKeras.
-	Default parameters for the LightGBM algorithm within AutoKeras are specified directly in the AutoKeras code. This includes settings or defaults that are required by AutoKeras for proper integration.
-	The required_preprocessing list specifies the preprocessing steps needed for both binary and multiclass classification, ensuring proper data preprocessing compatibility with AutoKeras.
-	The AlgorithmsRegistry.add statements are adjusted to reflect the necessary changes for AutoKeras, including the algorithm class, parameters, and defaults specific to the AutoKeras environment.

8.	ExtraTrees:
Here, we implemented a new classifier named AutoKerasClassifier that extends the BaseAlgorithm and ClassifierMixin classes. This classifier utilizes the AutoKeras library, specifically the StructuredDataClassifier for structured data classification tasks. 
-	max_iter_autokeras: Controls the maximum number of AutoKeras trials during model search.
-	epochs_autokeras: Specifies the number of epochs for training each model discovered by AutoKeras.
-	The __init__ method initializes the classifier with the specified parameters, and the fit method trains the model using the fit function from AutoKeras.
-	The is_fitted method checks whether the model has been fitted or not.
-	The predict method allows making predictions using the trained AutoKeras model.
-	The AutoKerasClassifier is registered with the AlgorithmsRegistry for binary classification tasks (BINARY_CLASSIFICATION).
 
9.	KNN:
-	We have two new classes : AutoKerasClassifierAlgorithm and AutoKerasRegressorAlgorithm, both inheriting from BaseAlgorithm and implementing ClassifierMixin and RegressorMixin respectively. These classes utilize the AutoKeras library for automatic model selection and hyperparameter tuning. 
-	Inherits from BaseAlgorithm and implements ClassifierMixin.
-	algorithm_name and algorithm_short_name are set to "AutoKeras" for classification tasks.
-	The constructor (__init__) initializes the model using StructuredDataClassifier from AutoKeras with specific parameters such as max_trials and epochs.
 
-	Inherits from BaseAlgorithm and implements RegressorMixin.
-	algorithm_name and algorithm_short_name are set to "AutoKeras" for regression tasks.
-	The constructor (__init__) initializes the model using StructuredDataRegressor from AutoKeras with specific parameters such as max_trials and epochs.
-	fit() Method:
The fit method for both classifier and regressor classes fits the AutoKeras model to the provided data (X and y). It uses a fixed number of epochs for training.
-	is_fitted() and predict() Methods:
The is_fitted method checks if the model has been fitted by verifying the existence of the model attribute.
The predict method reloads the model (if needed) and performs predictions on new data (X).
10.	CatBoost

-	The AutoKerasClassifierAlgorithm class is introduced to handle binary classification tasks using AutoKeras.
-	It inherits from BaseAlgorithm and implements ClassifierMixin.
-	The constructor (__init__) initializes the model using AutoKeras's AutoModel for binary classification.
-	The AutoKerasRegressorAlgorithm class is created for regression tasks using AutoKeras.
-	It inherits from BaseAlgorithm and implements RegressorMixin.
-	The constructor (__init__) initializes the model using AutoKeras's AutoModel for regression.
-	The AutoKerasAlgorithm class serves as a generic class for AutoKeras, handling all ML tasks (binary classification, multiclass classification, and regression).
-	It inherits from BaseAlgorithm.
-	The constructor (__init__) sets up the model based on the ML task specified in the parameters.
-	The fit method trains the AutoKeras model using the provided data.
-	The AutoKeras classifier and regressor algorithms are registered in the AlgorithmsRegistry for binary classification, multiclass classification, and regression tasks.
-	The registration includes the new AutoKeras classes, required preprocessing steps, and additional parameters.

-	Parameters specific to AutoKeras, such as max_trials and epochs, are set within the AutoKeras algorithm classes during initialization.
