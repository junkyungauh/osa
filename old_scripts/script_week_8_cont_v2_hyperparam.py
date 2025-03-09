import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize  # For weight optimization
from models.models import (
    LassoWrapper, RandomForestWrapper, XGBoostWrapper,
    StackingRegressorWrapper, ensemble_weighted_average, optimize_model_hyperparameters, optimize_arima, optimize_lstm
)
from preprocessing.preprocessing import normalize_data, create_lagged_features, split_temporal_data
from metrics.metrics import calculate_regression_metrics
from kalman_filter.kalman_filter import (
    ConstantVelocityKalmanFilter, FinancialModelKalmanFilter, optimize_kalman_hyperparameters
)

# Hyperparameters and Configurations
RANDOM_STATE = 42
WINDOW_SIZE = 10
LASSO_PARAM_GRID = {"alpha": np.logspace(-6, 2, 50)}
RF_PARAM_GRID = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0]
}
STACKING_META_PARAM_GRID = {"alpha": np.logspace(-6, 2, 10)}
CVKF_PARAM_GRID = [
    {"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r]}
    for q in [0.01, 0.1, 1.0]
    for r in [0.01, 0.1, 1.0]
]
FMKF_PARAM_GRID = [
    {"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r], "alpha": [a], "beta": [b]}
    for q in [0.01, 0.1, 1.0]
    for r in [0.01, 0.1, 1.0]
    for a in [0.6, 0.8, 1.0]
    for b in [0.1, 0.2, 0.4]
]

# File paths
cont_data_path = 'simulated_series_cont.csv'

# 1. Five-Way Split Function
def five_way_split(X, y, val1_size=0.2, val2_size=0.1, kalman_size=0.1, test_size=0.2):
    """Split data into train, validation (val1), ensemble-validation (val2), Kalman filter training, and test sets."""
    total_len = len(X)
    test_len = int(total_len * test_size)
    kalman_len = int(total_len * kalman_size)
    val2_len = int(total_len * val2_size)
    val1_len = int(total_len * val1_size)

    train_idx = range(0, total_len - test_len - kalman_len - val2_len - val1_len)
    val1_idx = range(total_len - test_len - kalman_len - val2_len - val1_len, total_len - test_len - kalman_len - val2_len)
    val2_idx = range(total_len - test_len - kalman_len - val2_len, total_len - test_len - kalman_len)
    kalman_idx = range(total_len - test_len - kalman_len, total_len - test_len)
    test_idx = range(total_len - test_len, total_len)

    return (
        X.iloc[train_idx], X.iloc[val1_idx], X.iloc[val2_idx], X.iloc[kalman_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[val1_idx], y.iloc[val2_idx], y.iloc[kalman_idx], y.iloc[test_idx]
    )

# 2. Load and Preprocess Data
def preprocess_data(file_path, target_column, lag_features=None, normalize=True):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    if lag_features:
        for column, lag in lag_features.items():
            data = create_lagged_features(data, column, lag)
    if normalize:
        data = normalize_data(data)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Preprocess regression data
X_cont, y_cont = preprocess_data(file_path=cont_data_path, target_column="return", lag_features={"return": 1})

# Split data into train, val1, val2, Kalman, and test
X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(X_cont, y_cont)

# Debugging: Ensure splits are valid
print("Train range:", X_train.index.min(), "-", X_train.index.max())
print("Validation 1 range:", X_val1.index.min(), "-", X_val1.index.max())
print("Validation 2 range:", X_val2.index.min(), "-", X_val2.index.max())
print("Kalman range:", X_kalman.index.min(), "-", X_kalman.index.max())
print("Test range:", X_test.index.min(), "-", X_test.index.max())

### Baselines ###
# T-1 Baseline
t_minus_1_predictions = X_test["return_lag1"]
t_minus_1_metrics = calculate_regression_metrics(y_test, t_minus_1_predictions)
print("T-1 Baseline Metrics:", t_minus_1_metrics)

# Windowed Average Baseline
def calculate_windowed_average(series, window_size):
    """Calculate rolling average as a baseline."""
    return series.rolling(window=window_size).mean().shift(1)

y_windowed_avg_test = calculate_windowed_average(pd.concat([y_train, y_val1, y_val2, y_test]), WINDOW_SIZE).iloc[len(y_train) + len(y_val1) + len(y_val2):]
windowed_avg_metrics = calculate_regression_metrics(y_test, y_windowed_avg_test.dropna())
print(f"Windowed Average (Window={WINDOW_SIZE}) Metrics:", windowed_avg_metrics)

### Base Models ###
# Lasso Hyperparameter Tuning
lasso_base_model, _ = optimize_model_hyperparameters(
    LassoWrapper, LASSO_PARAM_GRID, X_train, y_train, validation_data=(X_val1, y_val1)
)

# Random Forest Hyperparameter Tuning
rf_model, _ = optimize_model_hyperparameters(
    RandomForestWrapper, RF_PARAM_GRID, X_train, y_train, validation_data=(X_val1, y_val1)
)

# XGBoost Hyperparameter Tuning
xgb_model, _ = optimize_model_hyperparameters(
    XGBoostWrapper, XGB_PARAM_GRID, X_train, y_train, validation_data=(X_val1, y_val1)
)

# ARIMA and LSTM Models
arima_model = optimize_arima(y_train=y_train, p_values=[0, 1, 2], d_values=[0, 1], q_values=[0, 1, 2])
arima_predictions = arima_model.predict(start=len(y_train) + len(y_val1), end=len(y_train) + len(y_val1) + len(y_test) - 1)
arima_metrics = calculate_regression_metrics(y_test, arima_predictions)

lstm_model = optimize_lstm(
    X_train=X_train.values.reshape(-1, 1, X_train.shape[1]),
    y_train=y_train.values,
    input_size=X_train.shape[1],
    hidden_sizes=[50, 100],
    learning_rates=[0.01, 0.1],
    num_epochs_list=[10, 20]
)
lstm_predictions = lstm_model.predict(X_test.values.reshape(-1, 1, X_test.shape[1]))
lstm_metrics = calculate_regression_metrics(y_test, lstm_predictions)

### Ensemble Models ###
# Generate predictions for each base model
lasso_predictions = lasso_base_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

ensemble_predictions = np.array([lasso_predictions, rf_predictions, xgb_predictions, arima_predictions, lstm_predictions])

# 1. Stacking Ensemble
stacking_features_train = np.column_stack([model.predict(X_train) for model in [lasso_base_model, rf_model, xgb_model]])
stacking_features_val2 = np.column_stack([model.predict(X_val2) for model in [lasso_base_model, rf_model, xgb_model]])
meta_param_grid = {"alpha": np.logspace(-6, 2, 10)}
lasso_meta_model_stacking, _ = optimize_model_hyperparameters(
    LassoWrapper, meta_param_grid, stacking_features_train, y_train, validation_data=(stacking_features_val2, y_val2)
)
stacking_model = StackingRegressorWrapper(base_models=[
    ("lasso", lasso_base_model),
    ("rf", rf_model),
    ("xgb", xgb_model)
], meta_model=lasso_meta_model_stacking)
stacking_model.fit(X_train, y_train)
stacking_predictions = stacking_model.predict(X_test)
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)
print("Stacking Ensemble Metrics:", stacking_metrics)

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
lasso_ensemble_features_train = np.column_stack([model.predict(X_train) for model in [lasso_base_model, rf_model, xgb_model]])
lasso_ensemble_features_val2 = np.column_stack([model.predict(X_val2) for model in [lasso_base_model, rf_model, xgb_model]])
lasso_ensemble_model, _ = optimize_model_hyperparameters(
    LassoWrapper, lasso_param_grid, lasso_ensemble_features_train, y_train, validation_data=(lasso_ensemble_features_val2, y_val2)
)
lasso_ensemble_features_test = np.column_stack([model.predict(X_test) for model in [lasso_base_model, rf_model, xgb_model]])
lasso_ensemble_predictions = lasso_ensemble_model.predict(lasso_ensemble_features_test)
lasso_ensemble_metrics = calculate_regression_metrics(y_test, lasso_ensemble_predictions)
print("Lasso Ensemble Metrics:", lasso_ensemble_metrics)

### Kalman Filter Integration for Ensembles ###
# Ensure stacking_predictions_trimmed is the correct shape
stacking_predictions_trimmed = stacking_predictions[:len(y_test)]
print(f"Trimmed stacking predictions: {stacking_predictions_trimmed.shape}, y_test: {y_test.shape}")

# Apply Constant Velocity Kalman Filter
cvkf_params, cvkf_preds = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    [{"initial_state": np.array([stacking_predictions_trimmed[0]]), "Q_diag": [q], "R_diag": [r]}
     for q in [0.01, 0.1, 1.0]
     for r in [0.01, 0.1, 1.0]],
    stacking_predictions_trimmed,
    y_test
)

# Flatten and align cvkf_preds
cvkf_preds = np.array(cvkf_preds).squeeze()  # Ensure shape is 1D
if cvkf_preds.shape[0] != y_test.shape[0]:
    raise ValueError(f"Shape mismatch after CVKF: cvkf_preds={cvkf_preds.shape}, y_test={y_test.shape}")

cvkf_metrics = calculate_regression_metrics(y_test, cvkf_preds)
print("Metrics with CVKF (Corrected):", cvkf_metrics)

# Apply Financial Model Kalman Filter
fmkf_params, fmkf_preds = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    [{"initial_state": np.array([stacking_predictions_trimmed[0]]), "Q_diag": [q], "R_diag": [r], "alpha": [a], "beta": [b]}
     for q in [0.01, 0.1, 1.0]
     for r in [0.01, 0.1, 1.0]
     for a in [0.6, 0.8, 1.0]
     for b in [0.1, 0.2, 0.4]],
    stacking_predictions_trimmed,
    y_test
)

# Flatten and align fmkf_preds
fmkf_preds = np.array(fmkf_preds).squeeze()  # Ensure shape is 1D
if fmkf_preds.shape[0] != y_test.shape[0]:
    raise ValueError(f"Shape mismatch after FMKF: fmkf_preds={fmkf_preds.shape}, y_test={y_test.shape}")

fmkf_metrics = calculate_regression_metrics(y_test, fmkf_preds)
print("Metrics with FMKF (Corrected):", fmkf_metrics)



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
    "Lasso Ensemble": lasso_ensemble_metrics,
    "CVKF (Stacking)": cvkf_metrics,
    "FMKF (Stacking)": fmkf_metrics
}

metrics_df = pd.DataFrame(all_metrics).T
print("Final Model Metrics with Ensembles and Kalman Filters:\n", metrics_df)

# Visualize metrics for comparison
metrics_df[["MAE", "RMSE"]].plot(kind="bar", figsize=(14, 7), title="Model Comparison Metrics (With Kalman Filters and Ensembles)")
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
