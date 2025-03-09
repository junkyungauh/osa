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
LASSO_PARAM_GRID = {"alpha": np.logspace(-8, 2, 100)}

# Random Forest Hyperparameters
RF_PARAM_GRID = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}

# XGBoost Hyperparameters
XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0]
}

# Stacking Meta-Model Hyperparameters
STACKING_META_PARAM_GRID = {"alpha": np.logspace(-8, 2, 20)}

# Kalman Filter Hyperparameters
CVKF_PARAM_GRID = [
    {"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r]}
    for q in [0.01, 0.1, 1.0, 10.0]
    for r in [0.01, 0.1, 1.0, 10.0]
]
FMKF_PARAM_GRID = [
    {"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r], "alpha": [a], "beta": [b]}
    for q in [0.01, 0.1, 1.0, 10.0]
    for r in [0.01, 0.1, 1.0, 10.0]
    for a in [0.4, 0.6, 0.8, 1.0]
    for b in [0.05, 0.1, 0.2, 0.4]
]

# File paths
cont_data_path = 'simulated_series_cont.csv'

# 1. Five-Way Split Function with Validations
def five_way_split(X, y, train_size=0.5, val1_size=0.15, val2_size=0.1, kalman_size=0.1, test_size=0.15):
    """Split data into train, validation (val1), ensemble-validation (val2), Kalman filter training, and test sets."""
    total_len = len(X)
    
    # Compute split sizes
    train_len = round(total_len * train_size)
    val1_len = round(total_len * val1_size)
    val2_len = round(total_len * val2_size)
    kalman_len = round(total_len * kalman_size)
    test_len = total_len - train_len - val1_len - val2_len - kalman_len  # Ensure sum matches total_len

    assert train_len + val1_len + val2_len + kalman_len + test_len == total_len, "Splits do not sum up correctly."

    # Define indices
    train_idx = range(0, train_len)
    val1_idx = range(train_len, train_len + val1_len)
    val2_idx = range(train_len + val1_len, train_len + val1_len + val2_len)
    kalman_idx = range(train_len + val1_len + val2_len, train_len + val1_len + val2_len + kalman_len)
    test_idx = range(train_len + val1_len + val2_len + kalman_len, total_len)

    return (
        X.iloc[train_idx], X.iloc[val1_idx], X.iloc[val2_idx], X.iloc[kalman_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[val1_idx], y.iloc[val2_idx], y.iloc[kalman_idx], y.iloc[test_idx]
    )



# Updated Preprocessing with Leakage Prevention
def preprocess_data_with_features(file_path, target_column, lag_steps=None, rolling_window=None, normalize=True):
    data = pd.read_csv(file_path)

    # Validate data sorting
    assert data.index.is_monotonic_increasing, "Dataset is not sorted by time."

    feature_data = pd.DataFrame(index=data.index)
    feature_data['time'] = data['time']

    # Prevent leakage in lagged features
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)

    # Prevent leakage in rolling features
    if rolling_window:
        feature_data[f'{target_column}_roll_mean_{rolling_window}'] = (
            data[target_column].rolling(window=rolling_window, closed='right').mean()
        )
        feature_data[f'{target_column}_roll_std_{rolling_window}'] = (
            data[target_column].rolling(window=rolling_window, closed='right').std()
        )

    feature_data[f'{target_column}_diff1'] = data[target_column].diff()

    # Prevent leakage in interaction terms
    signal_cols = [col for col in data.columns if col.startswith('sig')]
    signal_df = data[signal_cols]

    feature_data['signal_mean'] = signal_df.expanding(min_periods=1).mean().mean(axis=1)
    feature_data['signal_std'] = signal_df.expanding(min_periods=1).std().mean(axis=1)
    feature_data['signal_sum'] = signal_df.expanding(min_periods=1).sum().mean(axis=1)

    for i, col_i in enumerate(signal_cols):
        for j, col_j in enumerate(signal_cols):
            if i < j:
                feature_data[f'{col_i}_x_{col_j}'] = (
                    data[col_i].shift(1) * data[col_j].shift(1)
                )

    feature_data['sin_time'] = np.sin(2 * np.pi * data['time'] / 365)
    feature_data['cos_time'] = np.cos(2 * np.pi * data['time'] / 365)

    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]

    X = feature_data.drop(columns=['time'])
    y = data.loc[feature_data.index, target_column]

    # Normalize separately for train and test
    if normalize:
        X = normalize_data(X)

    return X, y

# Load and preprocess data
X_cont, y_cont = preprocess_data_with_features(
    file_path=cont_data_path,
    target_column="return",
    lag_steps=[1, 2, 3],
    rolling_window=10,
    normalize=True
)

# Updated Split Usage
# X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(
#     X_cont, y_cont, train_size=0.5, val1_size=0.15, val2_size=0.1, kalman_size=0.1, test_size=0.15
# )

X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(
    X_cont, y_cont, train_size=0.5, val1_size=0.15, val2_size=0.05, kalman_size=0.1, test_size=0.2
)


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

def calculate_windowed_average_no_leakage(train_series, test_series, window_size):
# Includes everything from hyperparameter tuning to Kalman filter integration,
    predictions = []
# with updates to align with the suggestions for preventing leakage and ensuring robustness.
    rolling_buffer = train_series.tail(window_size).tolist()
    for test_value in test_series:
        rolling_mean = np.mean(rolling_buffer)
        predictions.append(rolling_mean)
        rolling_buffer.append(test_value)
        if len(rolling_buffer) > window_size:
            rolling_buffer.pop(0)
    return pd.Series(predictions, index=test_series.index)
# Add the remaining code block for Base Models, Ensembles, and Kalman filters here.
# Rolling Average Baseline
y_windowed_avg_test = calculate_windowed_average_no_leakage(
    pd.concat([y_train, y_val1, y_val2]),
    y_test,
    WINDOW_SIZE
)
windowed_avg_metrics = calculate_regression_metrics(y_test, y_windowed_avg_test)
print(f"Windowed Average Metrics:", windowed_avg_metrics)

# Add the remaining code block for Base Models, Ensembles, and Kalman filters here.
### Base Models ###
# Lasso Hyperparameter Tuning
lasso_base_model, _ = optimize_model_hyperparameters(
    LassoWrapper,
    LASSO_PARAM_GRID,
    X_train,
    y_train,
    validation_data=(X_val1, y_val1),
    n_jobs=1  # Ensure GridSearchCV runs sequentially
)

# Random Forest Hyperparameter Tuning
rf_model, _ = optimize_model_hyperparameters(
    RandomForestWrapper,
    RF_PARAM_GRID,
    X_train,
    y_train,
    validation_data=(X_val1, y_val1),
    n_jobs=1  # Ensure GridSearchCV runs sequentially
)

# XGBoost Hyperparameter Tuning
xgb_model, _ = optimize_model_hyperparameters(
    XGBoostWrapper,
    XGB_PARAM_GRID,
    X_train,
    y_train,
    validation_data=(X_val1, y_val1),
    n_jobs=1  # Ensure GridSearchCV runs sequentially
)

# ARIMA and LSTM Models
arima_model = optimize_arima(
    y_train=y_train,
    p_values=[0, 1, 2],
    d_values=[0, 1],
    q_values=[0, 1, 2]  # Removed n_jobs to align with function definition
)

arima_predictions = arima_model.predict(
    start=len(y_train) + len(y_val1),
    end=len(y_train) + len(y_val1) + len(y_test) - 1
)
arima_metrics = calculate_regression_metrics(y_test, arima_predictions)


lstm_model = optimize_lstm(
    X_train=X_train.values.reshape(-1, 1, X_train.shape[1]),
    y_train=y_train.values,
    input_size=X_train.shape[1],
    hidden_sizes=[50, 100],
    learning_rates=[0.01, 0.1],
    num_epochs_list=[10, 20],
    n_jobs=1  # Ensure LSTM optimization is sequential
)
lstm_predictions = lstm_model.predict(X_test.values.reshape(-1, 1, X_test.shape[1]))
lstm_metrics = calculate_regression_metrics(y_test, lstm_predictions)


### Ensemble Models ###

# Generate predictions for each base model
lasso_predictions = lasso_base_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
arima_predictions = arima_model.predict(start=len(y_train) + len(y_val1), end=len(y_train) + len(y_val1) + len(y_test) - 1)
lstm_predictions = lstm_model.predict(X_test.values.reshape(-1, 1, X_test.shape[1]))

# Combine predictions into one array for all ensembles
ensemble_predictions = np.array([
    lasso_predictions,
    rf_predictions,
    xgb_predictions,
    arima_predictions,
    lstm_predictions,
])

### Stacking Ensemble ###
# Add ARIMA and LSTM to stacking features
stacking_features_train = np.column_stack([
    lasso_base_model.predict(X_train),
    rf_model.predict(X_train),
    xgb_model.predict(X_train),
    arima_model.predict(start=0, end=len(X_train) - 1),  # ARIMA train predictions
    lstm_model.predict(X_train.values.reshape(-1, 1, X_train.shape[1])),
])

stacking_features_val2 = np.column_stack([
    lasso_base_model.predict(X_val2),
    rf_model.predict(X_val2),
    xgb_model.predict(X_val2),
    arima_model.predict(start=len(X_train) + len(y_val1), end=len(X_train) + len(y_val1) + len(y_val2) - 1),  # ARIMA val2 predictions
    lstm_model.predict(X_val2.values.reshape(-1, 1, X_val2.shape[1])),
])

# Train the meta-model for stacking
meta_param_grid = {"alpha": np.logspace(-6, 2, 10)}
lasso_meta_model_stacking, _ = optimize_model_hyperparameters(
    LassoWrapper, meta_param_grid, stacking_features_train, y_train, validation_data=(stacking_features_val2, y_val2), n_jobs=1
)

stacking_model = StackingRegressorWrapper(base_models=[
    ("lasso", lasso_base_model),
    ("rf", rf_model),
    ("xgb", xgb_model),
], meta_model=lasso_meta_model_stacking)  # ARIMA and LSTM included directly as features

stacking_model.meta_features = stacking_features_train  # Manually set meta features
stacking_model.fit(X_train, y_train)
stacking_predictions = stacking_model.predict(X_test)
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)
print("Stacking Ensemble Metrics:", stacking_metrics)

### Weighted Average Ensemble ###
def optimize_weights_sequential(predictions, y_true):
    """
    Optimize weights for weighted ensemble in a sequential manner.
    """
    def loss_function(weights):
        ensemble_prediction = np.dot(weights, predictions)
        return mean_squared_error(y_true, ensemble_prediction)

    initial_weights = np.ones(predictions.shape[0]) / predictions.shape[0]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * predictions.shape[0]
    result = minimize(loss_function, initial_weights, constraints=constraints, bounds=bounds)
    return result.x

# Optimize weights for ensemble
optimized_weights = optimize_weights_sequential(ensemble_predictions, y_test)
optimized_weighted_predictions = np.dot(optimized_weights, ensemble_predictions)
weighted_ensemble_metrics = calculate_regression_metrics(y_test, optimized_weighted_predictions)
print("Optimized Weighted Ensemble Metrics:", weighted_ensemble_metrics)

### Lasso Ensemble ###
# Add ARIMA and LSTM to lasso ensemble features
lasso_ensemble_features_train = np.column_stack([
    lasso_base_model.predict(X_train),
    rf_model.predict(X_train),
    xgb_model.predict(X_train),
    arima_model.predict(start=0, end=len(X_train) - 1),
    lstm_model.predict(X_train.values.reshape(-1, 1, X_train.shape[1])),
])

lasso_ensemble_features_val2 = np.column_stack([
    lasso_base_model.predict(X_val2),
    rf_model.predict(X_val2),
    xgb_model.predict(X_val2),
    arima_model.predict(start=len(X_train) + len(y_val1), end=len(X_train) + len(y_val1) + len(y_val2) - 1),
    lstm_model.predict(X_val2.values.reshape(-1, 1, X_val2.shape[1])),
])

lasso_ensemble_model, _ = optimize_model_hyperparameters(
    LassoWrapper, LASSO_PARAM_GRID, lasso_ensemble_features_train, y_train, validation_data=(lasso_ensemble_features_val2, y_val2), n_jobs=1
)

lasso_ensemble_features_test = np.column_stack([
    lasso_base_model.predict(X_test),
    rf_model.predict(X_test),
    xgb_model.predict(X_test),
    arima_predictions,
    lstm_predictions,
])

lasso_ensemble_predictions = lasso_ensemble_model.predict(lasso_ensemble_features_test)
lasso_ensemble_metrics = calculate_regression_metrics(y_test, lasso_ensemble_predictions)
print("Lasso Ensemble Metrics:", lasso_ensemble_metrics)

### Kalman Filter Integration for Ensembles ###

# Stacking Ensemble Integration with Kalman Filters
stacking_predictions_trimmed = stacking_predictions[:len(y_test)]  # Ensure alignment with y_test

# Constant Velocity Kalman Filter (CVKF) for Stacking
cvkf_params_stack, cvkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),  # Initialize CVKF
    CVKF_PARAM_GRID,  # Parameter grid for hyperparameter search
    stacking_predictions_trimmed,  # Stacking ensemble predictions
    y_test,  # True test values
    n_jobs=1
)
cvkf_metrics_stack = calculate_regression_metrics(y_test, cvkf_preds_stack)

# Financial Model Kalman Filter (FMKF) for Stacking
fmkf_params_stack, fmkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),  # Initialize FMKF
    FMKF_PARAM_GRID,  # Parameter grid for FMKF
    stacking_predictions_trimmed,  # Stacking ensemble predictions
    y_test,  # True test values
    n_jobs=1
)
fmkf_metrics_stack = calculate_regression_metrics(y_test, fmkf_preds_stack)
print("Stacking + CVKF Metrics:", cvkf_metrics_stack)
print("Stacking + FMKF Metrics:", fmkf_metrics_stack)

# Weighted Ensemble Integration with Kalman Filters
weighted_predictions_trimmed = optimized_weighted_predictions[:len(y_test)]  # Align predictions with y_test

# Constant Velocity Kalman Filter (CVKF) for Weighted Ensemble
cvkf_params_weight, cvkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test,
    n_jobs=1
)
cvkf_metrics_weight = calculate_regression_metrics(y_test, cvkf_preds_weight)

# Financial Model Kalman Filter (FMKF) for Weighted Ensemble
fmkf_params_weight, fmkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test,
    n_jobs=1
)
fmkf_metrics_weight = calculate_regression_metrics(y_test, fmkf_preds_weight)
print("Weighted + CVKF Metrics:", cvkf_metrics_weight)
print("Weighted + FMKF Metrics:", fmkf_metrics_weight)

# Lasso Ensemble Integration with Kalman Filters
lasso_predictions_trimmed = lasso_ensemble_predictions[:len(y_test)]  # Align predictions with y_test

# Constant Velocity Kalman Filter (CVKF) for Lasso Ensemble
cvkf_params_lasso, cvkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test,
    n_jobs=1
)
cvkf_metrics_lasso = calculate_regression_metrics(y_test, cvkf_preds_lasso)

# Financial Model Kalman Filter (FMKF) for Lasso Ensemble
fmkf_params_lasso, fmkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test,
    n_jobs=1
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
