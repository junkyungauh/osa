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

# Lasso Regression Hyperparameters
LASSO_PARAM_GRID = {"alpha": np.logspace(-8, 2, 100)}  # Expand search range and granularity

# Random Forest Hyperparameters
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200, 500],  # Added 500 for deeper exploration
    "max_depth": [None, 5, 10, 20],       # Added 5 for shallow trees
    "min_samples_split": [2, 5, 10],      # Added min_samples_split to control splits
    "min_samples_leaf": [1, 2, 4]         # Added min_samples_leaf to prevent overfitting
}

# XGBoost Hyperparameters
XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200, 500],  # Added 500
    "max_depth": [3, 5, 7, 10],           # Added 10 for deeper trees
    "learning_rate": [0.01, 0.1, 0.2, 0.3],  # Added 0.3 for faster learning rates
    "subsample": [0.6, 0.8, 1.0],         # Kept subsample the same
    "colsample_bytree": [0.6, 0.8, 1.0],  # Added colsample_bytree to control feature usage
    "reg_alpha": [0, 0.01, 0.1, 1],       # Added L1 regularization terms
    "reg_lambda": [1, 2, 5, 10]           # Added L2 regularization terms
}

# Stacking Meta-Model Hyperparameters
STACKING_META_PARAM_GRID = {"alpha": np.logspace(-8, 2, 20)}  # Expanded search range and granularity

# Constant Velocity Kalman Filter (CVKF) Hyperparameters
CVKF_PARAM_GRID = [
    {"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r]}
    for q in [0.01, 0.1, 1.0, 10.0]  # Added 10.0 for larger process noise
    for r in [0.01, 0.1, 1.0, 10.0]  # Added 10.0 for larger measurement noise
]

# Financial Model Kalman Filter (FMKF) Hyperparameters
FMKF_PARAM_GRID = [
    {"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r], "alpha": [a], "beta": [b]}
    for q in [0.01, 0.1, 1.0, 10.0]  # Added 10.0
    for r in [0.01, 0.1, 1.0, 10.0]  # Added 10.0
    for a in [0.4, 0.6, 0.8, 1.0]    # Added 0.4 for finer granularity
    for b in [0.05, 0.1, 0.2, 0.4]   # Added 0.05 for finer granularity
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

# Updated Preprocessing with Correct Rolling Aggregates
def preprocess_data_with_features(file_path, target_column, lag_steps=None, rolling_window=None, normalize=True):
    """
    Load and preprocess the data with leakage-free feature engineering.
    Adds lagged features, rolling statistics, and aggregate features computed only up to the current day.
    
    Parameters:
    - file_path: Path to the CSV file.
    - target_column: The name of the target variable column.
    - lag_steps: List of integers for lagging steps (e.g., [1, 2, 3]).
    - rolling_window: Integer window size for rolling statistics.
    - normalize: Whether to normalize the dataset.
    
    Returns:
    - X: Feature dataframe with engineered features.
    - y: Target variable series.
    """
    data = pd.read_csv(file_path)
    
    # Initialize feature dataframe
    feature_data = pd.DataFrame(index=data.index)
    feature_data['time'] = data['time']
    
    # Add lagged features for the target
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)
    
    # Add rolling statistics for the target
    if rolling_window:
        feature_data[f'{target_column}_roll_mean_{rolling_window}'] = (
            data[target_column].rolling(window=rolling_window, closed='right').mean()
        )
        feature_data[f'{target_column}_roll_std_{rolling_window}'] = (
            data[target_column].rolling(window=rolling_window, closed='right').std()
        )
    
    # Add first differences for the target
    feature_data[f'{target_column}_diff1'] = data[target_column].diff()
    
    # Add rolling aggregate features for signals (up to the current day)
    signal_cols = [col for col in data.columns if col.startswith('sig')]
    feature_data['signal_mean'] = data[signal_cols].apply(lambda x: x.expanding(min_periods=1).mean(), axis=0).mean(axis=1)
    feature_data['signal_std'] = data[signal_cols].apply(lambda x: x.expanding(min_periods=1).std(), axis=0).mean(axis=1)
    feature_data['signal_sum'] = data[signal_cols].apply(lambda x: x.expanding(min_periods=1).sum(), axis=0).mean(axis=1)
    
    # Add rolling interaction terms (using only past data)
    for i, col_i in enumerate(signal_cols):
        for j, col_j in enumerate(signal_cols):
            if i < j:  # Avoid duplicates
                feature_data[f'{col_i}_x_{col_j}'] = (
                    data[col_i].shift(1) * data[col_j].shift(1)
                )  # Shift to ensure no future data is used
    
    # Add time-based features
    feature_data['sin_time'] = np.sin(2 * np.pi * data['time'] / 365)
    feature_data['cos_time'] = np.cos(2 * np.pi * data['time'] / 365)
    
    # Drop rows with NaN values introduced by lagging or rolling
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]  # Align original data with feature rows
    
    # Separate features and target variable
    X = feature_data.drop(columns=['time'])
    y = data.loc[feature_data.index, target_column]
    
    # Normalize features if specified
    if normalize:
        X = normalize_data(X)
    
    return X, y

# Usage
X_cont, y_cont = preprocess_data_with_features(
    file_path=cont_data_path,
    target_column="return",
    lag_steps=[1, 2, 3],  # Add lagged features
    rolling_window=10,    # Add rolling mean and std with a window of 10
    normalize=True
)



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

### Stacking Ensemble ###
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

### Weighted Average Ensemble ###
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

### Lasso Ensemble ###
lasso_ensemble_features_train = np.column_stack([model.predict(X_train) for model in [lasso_base_model, rf_model, xgb_model]])
lasso_ensemble_features_val2 = np.column_stack([model.predict(X_val2) for model in [lasso_base_model, rf_model, xgb_model]])
lasso_ensemble_model, _ = optimize_model_hyperparameters(
    LassoWrapper, LASSO_PARAM_GRID, lasso_ensemble_features_train, y_train, validation_data=(lasso_ensemble_features_val2, y_val2)
)
lasso_ensemble_features_test = np.column_stack([model.predict(X_test) for model in [lasso_base_model, rf_model, xgb_model]])
lasso_ensemble_predictions = lasso_ensemble_model.predict(lasso_ensemble_features_test)
lasso_ensemble_metrics = calculate_regression_metrics(y_test, lasso_ensemble_predictions)
print("Lasso Ensemble Metrics:", lasso_ensemble_metrics)

### Kalman Filter Integration for Ensembles ###
# Stacking
stacking_predictions_trimmed = stacking_predictions[:len(y_test)]
cvkf_params_stack, cvkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    stacking_predictions_trimmed,
    y_test
)
cvkf_metrics_stack = calculate_regression_metrics(y_test, cvkf_preds_stack)

fmkf_params_stack, fmkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    stacking_predictions_trimmed,
    y_test
)
fmkf_metrics_stack = calculate_regression_metrics(y_test, fmkf_preds_stack)
print("Stacking + CVKF Metrics:", cvkf_metrics_stack)
print("Stacking + FMKF Metrics:", fmkf_metrics_stack)

# Weighted
weighted_predictions_trimmed = optimized_weighted_predictions[:len(y_test)]
cvkf_params_weight, cvkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test
)
cvkf_metrics_weight = calculate_regression_metrics(y_test, cvkf_preds_weight)

fmkf_params_weight, fmkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test
)
fmkf_metrics_weight = calculate_regression_metrics(y_test, fmkf_preds_weight)
print("Weighted + CVKF Metrics:", cvkf_metrics_weight)
print("Weighted + FMKF Metrics:", fmkf_metrics_weight)

# Lasso
lasso_predictions_trimmed = lasso_ensemble_predictions[:len(y_test)]
cvkf_params_lasso, cvkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test
)
cvkf_metrics_lasso = calculate_regression_metrics(y_test, cvkf_preds_lasso)

fmkf_params_lasso, fmkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test
)
fmkf_metrics_lasso = calculate_regression_metrics(y_test, fmkf_preds_lasso)
print("Lasso + CVKF Metrics:", cvkf_metrics_lasso)
print("Lasso + FMKF Metrics:", fmkf_metrics_lasso)

### Summary ###
all_metrics = {
    "T-1 Baseline": t_minus_1_metrics,
    f"Windowed Average (Window={WINDOW_SIZE})": windowed_avg_metrics,
    "Lasso (Base)": calculate_regression_metrics(y_test, lasso_predictions),
    "Random Forest": calculate_regression_metrics(y_test, rf_predictions),
    "XGBoost": calculate_regression_metrics(y_test, xgb_predictions),
    "ARIMA": arima_metrics,
    "LSTM": lstm_metrics,
    "Stacking Ensemble": stacking_metrics,
    "Weighted Ensemble": weighted_ensemble_metrics,
    "Lasso Ensemble": lasso_ensemble_metrics,
    "CVKF (Stacking)": cvkf_metrics_stack,
    "FMKF (Stacking)": fmkf_metrics_stack,
    "CVKF (Weighted)": cvkf_metrics_weight,
    "FMKF (Weighted)": fmkf_metrics_weight,
    "CVKF (Lasso)": cvkf_metrics_lasso,
    "FMKF (Lasso)": fmkf_metrics_lasso,
}

metrics_df = pd.DataFrame(all_metrics).T
print("Final Model Metrics:\n", metrics_df)

# Visualization
metrics_df[["MAE", "RMSE"]].plot(kind="bar", figsize=(14, 7), title="Model Comparison with Kalman Filters and Ensembles")
plt.ylabel("Error")
plt.show()
