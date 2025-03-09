import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
from models.models import (
    LinearRegressionWrapper, LassoWrapper, LogisticRegressionWrapper, RandomForestWrapper, XGBoostWrapper,
    StackingRegressorWrapper, ensemble_weighted_average, optimize_model_hyperparameters, optimize_arima, optimize_lstm
)
from preprocessing.preprocessing import (
    normalize_data, create_lagged_features, split_temporal_data
)
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics

# File paths
cont_data_path = 'simulated_series_cont.csv'
bin_data_path = 'simulated_series_bin.csv'

# Step 1: Load and Preprocess Data
def preprocess_data(file_path, target_column, lag_features=None, normalize=True):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    
    # Create lagged features
    if lag_features:
        for column, lag in lag_features.items():
            data = create_lagged_features(data, column, lag)
    
    # Normalize the data
    if normalize:
        data = normalize_data(data)
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Preprocess regression data
X_cont, y_cont = preprocess_data(
    file_path=cont_data_path, 
    target_column="return", 
    lag_features={"return": 1}
)

# Split data into train/validation/test sets for regression
X_train, X_val, X_test, y_train, y_val, y_test = split_temporal_data(X_cont, y_cont, val_size=0.2, test_size=0.2)

# Step 2: Base Model Hyperparameter Tuning and Evaluation
# Lasso Hyperparameter Tuning
lasso_param_grid = {"alpha": np.logspace(-6, 2, 50)}
lasso_model, _ = optimize_model_hyperparameters(
    LassoWrapper, lasso_param_grid, X_train, y_train, validation_data=(X_val, y_val)
)

# Random Forest Hyperparameter Tuning
rf_param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
random_forest_model, _ = optimize_model_hyperparameters(
    RandomForestWrapper, rf_param_grid, X_train, y_train, validation_data=(X_val, y_val)
)

# XGBoost Hyperparameter Tuning
xgb_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0]
}
xgboost_model, _ = optimize_model_hyperparameters(
    XGBoostWrapper, xgb_param_grid, X_train, y_train, validation_data=(X_val, y_val)
)

# ARIMA Hyperparameter Tuning
arima_model = optimize_arima(
    y_train=y_train, 
    p_values=[0, 1, 2], 
    d_values=[0, 1], 
    q_values=[0, 1, 2]
)
arima_predictions = arima_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
arima_metrics = calculate_regression_metrics(y_test, arima_predictions)
print("ARIMA Metrics:", arima_metrics)

# LSTM Hyperparameter Tuning
lstm_model = optimize_lstm(
    X_train=X_train.values.reshape(-1, 1, X_train.shape[1]),
    y_train=y_train.values,
    input_size=X_train.shape[1],
    hidden_sizes=[50, 100],
    learning_rates=[0.01, 0.1],
    num_epochs_list=[10, 20],
)
lstm_predictions = lstm_model.predict(X_test.values.reshape(-1, 1, X_test.shape[1]))
lstm_metrics = calculate_regression_metrics(y_test, lstm_predictions)
print("LSTM Metrics:", lstm_metrics)

# Step 3: Ensemble Model Hyperparameter Tuning and Evaluation
# Stacking Ensemble
base_models = [
    ("lasso", lasso_model),
    ("rf", random_forest_model),
    ("xgb", xgboost_model)
]
meta_model = LinearRegressionWrapper()
stacking_model = StackingRegressorWrapper(base_models, meta_model)
stacking_model.fit(X_train, y_train)

stacking_predictions = stacking_model.predict(X_test)
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)
print("Stacking Regression Metrics:", stacking_metrics)


# Generate predictions for each base model
lasso_predictions = lasso_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)
xgb_predictions = xgboost_model.predict(X_test)

# ARIMA predictions (already defined in the script)
arima_predictions = arima_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# LSTM predictions (already defined in the script)
lstm_predictions = lstm_model.predict(X_test.values.reshape(-1, 1, X_test.shape[1]))

# Weighted Average Ensemble
ensemble_predictions = np.array([lasso_predictions, rf_predictions, xgb_predictions, arima_predictions, lstm_predictions])
weights = np.ones(ensemble_predictions.shape[0]) / ensemble_predictions.shape[0]
weighted_predictions = ensemble_weighted_average(ensemble_predictions, weights)

ensemble_metrics = calculate_regression_metrics(y_test, weighted_predictions)
print("Weighted Ensemble Metrics:", ensemble_metrics)


# Step 4: Baseline Models - T-1 and Windowed Average
# T-1 Baseline
t_minus_1_predictions = X_test["return_lag1"]  # Using the lagged return as a prediction
t_minus_1_metrics = calculate_regression_metrics(y_test, t_minus_1_predictions)
print("T-1 Baseline Metrics:", t_minus_1_metrics)

# Windowed Average Baseline
def calculate_windowed_average(series, window_size):
    """Calculate rolling average as a baseline."""
    return series.rolling(window=window_size).mean().shift(1)  # Shift to prevent data leakage

# Generate windowed average predictions
window_size = 10  # Example: 10-step rolling average
y_windowed_avg_train = calculate_windowed_average(y_train, window_size)
y_windowed_avg_test = calculate_windowed_average(pd.concat([y_train, y_test]), window_size).iloc[len(y_train):]
windowed_avg_metrics = calculate_regression_metrics(y_test, y_windowed_avg_test.dropna())
print(f"Windowed Average (Window={window_size}) Metrics:", windowed_avg_metrics)

# Step 5: Summary and Visualizations
# Ensure all predictions are made and metrics are calculated

# Lasso Predictions and Metrics
lasso_predictions = lasso_model.predict(X_test)
lasso_metrics = calculate_regression_metrics(y_test, lasso_predictions)
print("Lasso Metrics:", lasso_metrics)

# Random Forest Predictions and Metrics
rf_predictions = random_forest_model.predict(X_test)
rf_metrics = calculate_regression_metrics(y_test, rf_predictions)
print("Random Forest Metrics:", rf_metrics)

# XGBoost Predictions and Metrics
xgb_predictions = xgboost_model.predict(X_test)
xgb_metrics = calculate_regression_metrics(y_test, xgb_predictions)
print("XGBoost Metrics:", xgb_metrics)

# ARIMA Metrics (already defined in the script)
# arima_predictions = already generated
arima_metrics = calculate_regression_metrics(y_test, arima_predictions)
print("ARIMA Metrics:", arima_metrics)

# LSTM Metrics (already defined in the script)
# lstm_predictions = already generated
lstm_metrics = calculate_regression_metrics(y_test, lstm_predictions)
print("LSTM Metrics:", lstm_metrics)

# Stacking Predictions and Metrics
stacking_predictions = stacking_model.predict(X_test)
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)
print("Stacking Metrics:", stacking_metrics)

# Weighted Ensemble Metrics (already defined in the script)
# weighted_predictions = already generated
ensemble_metrics = calculate_regression_metrics(y_test, weighted_predictions)
print("Weighted Ensemble Metrics:", ensemble_metrics)

# Baseline Metrics (already defined in the script)
# t_minus_1_predictions and y_windowed_avg_test = already generated
t_minus_1_metrics = calculate_regression_metrics(y_test, t_minus_1_predictions)
windowed_avg_metrics = calculate_regression_metrics(y_test, y_windowed_avg_test.dropna())
print("T-1 Baseline Metrics:", t_minus_1_metrics)
print(f"Windowed Average (Window={window_size}) Metrics:", windowed_avg_metrics)

# Combine all metrics into the dictionary
all_metrics = {
    "Lasso": lasso_metrics,
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics,
    "ARIMA": arima_metrics,
    "LSTM": lstm_metrics,
    "Stacking": stacking_metrics,
    "Weighted Ensemble": ensemble_metrics,
    "T-1 Baseline": t_minus_1_metrics,
    f"Windowed Average (Window={window_size})": windowed_avg_metrics
}

# Convert metrics to DataFrame and print
metrics_df = pd.DataFrame(all_metrics).T
print("All Model Metrics with Baselines:\n", metrics_df)

# Plot results for comparison
metrics_df[["MAE", "RMSE"]].plot(kind="bar", figsize=(14, 7), title="Model Comparison Metrics (Regression) with Baselines")
plt.ylabel("Error")
plt.show()

### Save Artifacts ###

# import os
# import joblib  # Importing joblib for saving and loading models

# # Directory to save artifacts
# model_save_dir = "models_saved/"
# os.makedirs(model_save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# def save_artifact(obj, artifact_name):
#     """
#     Save a model or artifact to the predefined directory.
    
#     Parameters:
#     obj (object): The model or artifact to save.
#     artifact_name (str): The name of the file to save the artifact as.
#     """
#     save_path = os.path.join(model_save_dir, artifact_name)
#     joblib.dump(obj, save_path)
#     print(f"Artifact '{artifact_name}' saved at: {save_path}")

# # Step 2: Base Model Hyperparameter Tuning and Evaluation
# # Lasso Hyperparameter Tuning
# lasso_param_grid = {"alpha": np.logspace(-6, 2, 50)}
# lasso_model, _ = optimize_model_hyperparameters(
#     LassoWrapper, lasso_param_grid, X_train, y_train, validation_data=(X_val, y_val)
# )
# save_artifact(lasso_model, "lasso_model.pkl")

# # Random Forest Hyperparameter Tuning
# rf_param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
# random_forest_model, _ = optimize_model_hyperparameters(
#     RandomForestWrapper, rf_param_grid, X_train, y_train, validation_data=(X_val, y_val)
# )
# save_artifact(random_forest_model, "random_forest_model.pkl")

# # XGBoost Hyperparameter Tuning
# xgb_param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [3, 5, 7],
#     "learning_rate": [0.01, 0.1, 0.2],
#     "subsample": [0.6, 0.8, 1.0]
# }
# xgboost_model, _ = optimize_model_hyperparameters(
#     XGBoostWrapper, xgb_param_grid, X_train, y_train, validation_data=(X_val, y_val)
# )
# save_artifact(xgboost_model, "xgboost_model.pkl")

# # ARIMA Hyperparameter Tuning
# arima_model = optimize_arima(
#     y_train=y_train, 
#     p_values=[0, 1, 2], 
#     d_values=[0, 1], 
#     q_values=[0, 1, 2]
# )
# save_artifact(arima_model, "arima_model.pkl")

# # LSTM Hyperparameter Tuning
# lstm_model = optimize_lstm(
#     X_train=X_train.values.reshape(-1, 1, X_train.shape[1]),
#     y_train=y_train.values,
#     input_size=X_train.shape[1],
#     hidden_sizes=[50, 100],
#     learning_rates=[0.01, 0.1],
#     num_epochs_list=[10, 20],
# )
# save_artifact(lstm_model, "lstm_model.pkl")

# # Stacking Ensemble
# base_models = [
#     ("lasso", lasso_model),
#     ("rf", random_forest_model),
#     ("xgb", xgboost_model)
# ]
# meta_model = LinearRegressionWrapper()
# stacking_model = StackingRegressorWrapper(base_models, meta_model)
# stacking_model.fit(X_train, y_train)
# save_artifact(stacking_model, "stacking_model.pkl")

# # Weighted Average Ensemble
# ensemble_predictions = np.array([lasso_predictions, rf_predictions, xgb_predictions, arima_predictions, lstm_predictions])
# weights = np.ones(ensemble_predictions.shape[0]) / ensemble_predictions.shape[0]
# weighted_predictions = ensemble_weighted_average(ensemble_predictions, weights)

# save_artifact(weights, "weighted_ensemble_weights.pkl")
# ensemble_metrics = calculate_regression_metrics(y_test, weighted_predictions)
# print("Weighted Ensemble Metrics:", ensemble_metrics)
