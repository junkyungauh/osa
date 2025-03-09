import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # For weight optimization
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

# Debugging: Ensure splits are valid
print("Train range:", X_train.index.min(), "-", X_train.index.max())
print("Validation range:", X_val.index.min(), "-", X_val.index.max())
print("Test range:", X_test.index.min(), "-", X_test.index.max())

### Baselines ###
# T-1 Baseline
t_minus_1_predictions = X_test["return_lag1"]  # Using the lagged return as a prediction
t_minus_1_metrics = calculate_regression_metrics(y_test, t_minus_1_predictions)
print("T-1 Baseline Metrics:", t_minus_1_metrics)

# Windowed Average Baseline
def calculate_windowed_average(series, window_size):
    """Calculate rolling average as a baseline."""
    return series.rolling(window=window_size).mean().shift(1)  # Shift to prevent data leakage

window_size = 10  # Example: 10-step rolling average
y_windowed_avg_train = calculate_windowed_average(y_train, window_size)
y_windowed_avg_test = calculate_windowed_average(pd.concat([y_train, y_test]), window_size).iloc[len(y_train):]
windowed_avg_metrics = calculate_regression_metrics(y_test, y_windowed_avg_test.dropna())
print(f"Windowed Average (Window={window_size}) Metrics:", windowed_avg_metrics)

### Base Models ###
# Lasso Hyperparameter Tuning
lasso_param_grid = {"alpha": np.logspace(-6, 2, 50)}
lasso_base_model, _ = optimize_model_hyperparameters(
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

### Ensemble Models ###
# Generate predictions for each base model
lasso_predictions = lasso_base_model.predict(X_test)
rf_predictions = random_forest_model.predict(X_test)
xgb_predictions = xgboost_model.predict(X_test)

ensemble_predictions = np.array([lasso_predictions, rf_predictions, xgb_predictions, arima_predictions, lstm_predictions])

# 1. Stacking Ensemble
base_models = [
    ("lasso", lasso_base_model),
    ("rf", random_forest_model),
    ("xgb", xgboost_model)
]

# Generate stacking features from base models
stacking_features_train = np.column_stack([model.predict(X_train) for _, model in base_models])
stacking_features_val = np.column_stack([model.predict(X_val) for _, model in base_models])

# Hyperparameter optimization for the meta-model (Lasso)
meta_param_grid = {"alpha": np.logspace(-6, 2, 10)}
lasso_meta_model, _ = optimize_model_hyperparameters(
    LassoWrapper, meta_param_grid, X_train=stacking_features_train, y_train=y_train, validation_data=(stacking_features_val, y_val)
)

# Build and evaluate the stacking model
stacking_model = StackingRegressorWrapper(base_models, lasso_meta_model)
stacking_model.fit(X_train, y_train)
stacking_predictions = stacking_model.predict(X_test)
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)
print("Optimized Stacking Regression Metrics:", stacking_metrics)

# 2. Weighted Average Ensemble
def optimize_weights(predictions, y_true):
    def loss_function(weights):
        ensemble_prediction = np.dot(weights, predictions)
        return mean_squared_error(y_true, ensemble_prediction)
    initial_weights = np.ones(predictions.shape[0]) / predictions.shape[0]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * predictions.shape[0]
    result = minimize(loss_function, initial_weights, constraints=constraints, bounds=bounds)
    return result.x
optimized_weights = optimize_weights(ensemble_predictions, y_test)
optimized_weighted_predictions = np.dot(optimized_weights, ensemble_predictions)
weighted_ensemble_metrics = calculate_regression_metrics(y_test, optimized_weighted_predictions)
print("Optimized Weighted Ensemble Metrics:", weighted_ensemble_metrics)

# 3. Lasso Ensemble
lasso_ensemble_features_train = np.column_stack([model.predict(X_train) for _, model in base_models])
lasso_ensemble_features_val = np.column_stack([model.predict(X_val) for _, model in base_models])
lasso_ensemble_model, _ = optimize_model_hyperparameters(
    LassoWrapper, lasso_param_grid, lasso_ensemble_features_train, y_train, validation_data=(lasso_ensemble_features_val, y_val)
)
lasso_ensemble_features_test = np.column_stack([model.predict(X_test) for _, model in base_models])
lasso_ensemble_predictions = lasso_ensemble_model.predict(lasso_ensemble_features_test)
lasso_ensemble_metrics = calculate_regression_metrics(y_test, lasso_ensemble_predictions)
print("Lasso Ensemble Metrics:", lasso_ensemble_metrics)

### Summary ###
all_metrics = {
    "T-1 Baseline": t_minus_1_metrics,
    f"Windowed Average (Window={window_size})": windowed_avg_metrics,
    "Lasso (Base)": calculate_regression_metrics(y_test, lasso_predictions),
    "Random Forest": calculate_regression_metrics(y_test, rf_predictions),
    "XGBoost": calculate_regression_metrics(y_test, xgb_predictions),
    "ARIMA": arima_metrics,
    "LSTM": lstm_metrics,
    "Stacking Ensemble": stacking_metrics,
    "Weighted Ensemble": weighted_ensemble_metrics,
    "Lasso Ensemble": lasso_ensemble_metrics
}

metrics_df = pd.DataFrame(all_metrics).T
print("All Model Metrics with Ensembles:\n", metrics_df)

# Visualize metrics for comparison
metrics_df[["MAE", "RMSE"]].plot(kind="bar", figsize=(14, 7), title="Model Comparison Metrics (Regression) with Ensembles")
plt.ylabel("Error")
plt.show()

# Visualizing Ensemble Weights (Optional)
ensemble_weight_df = pd.DataFrame({
    "Model": ["Lasso", "Random Forest", "XGBoost", "ARIMA", "LSTM"],
    "Weight": optimized_weights
})
ensemble_weight_df.plot(
    kind="bar",
    x="Model",
    y="Weight",
    legend=False,
    title="Optimized Weights for Weighted Ensemble",
    figsize=(10, 6)
)
plt.ylabel("Weight")
plt.show()

   
