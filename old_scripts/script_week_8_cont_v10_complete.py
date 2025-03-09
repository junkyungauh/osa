import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # Ensure XGBoost is properly imported
from scipy.optimize import minimize  # For weight optimization
import torch  # Ensure PyTorch is imported
import torch.nn as nn
import torch.optim as optim

# Import models and utilities from the updated `model.py`
from models.models import (
    LassoWrapper, RandomForestWrapper, XGBoostWrapper,
    StackingRegressorWrapper, NeuralNetworkEnsembleWrapper,
    ensemble_weighted_average, optimize_model_hyperparameters, optimize_arima, optimize_lstm
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



# Updated Preprocessing Function
def preprocess_data_with_general_features(file_path, target_column, lag_steps=None, rolling_window=None, ema_window=None):
    data = pd.read_csv(file_path)

    # Validate data sorting
    assert data.index.is_monotonic_increasing, "Dataset is not sorted by time."
    assert rolling_window >= 3, "Rolling window size should be at least 3 to avoid capturing noise."

    feature_data = pd.DataFrame(index=data.index)
    # Feature Engineering (unchanged)
    signal_cols = [col for col in data.columns if col.startswith('sig')]
    for col in signal_cols:
        feature_data[f'{col}_roll_mean'] = data[col].rolling(window=rolling_window, closed='right').mean()
        feature_data[f'{col}_roll_std'] = data[col].rolling(window=rolling_window, closed='right').std()
    
    # Lagged Features
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)
    
    # Drop NaNs introduced by lagging
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]

    # Separate features and target
    X = feature_data
    y = data[target_column]

    return X, y



# Load and preprocess data
X_cont, y_cont = preprocess_data_with_general_features(
    file_path='simulated_series_cont.csv',
    target_column='return',
    lag_steps=[1, 2, 3],
    rolling_window=10,
    ema_window=5
)

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

# Rolling Average Baseline
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

y_windowed_avg_test = calculate_windowed_average_no_leakage(
    pd.concat([y_train, y_val1, y_val2]),
    y_test,
    WINDOW_SIZE
)
windowed_avg_metrics = calculate_regression_metrics(y_test, y_windowed_avg_test)
print(f"Windowed Average Metrics:", windowed_avg_metrics)


### Base Models ###
# Lasso Model with Pipeline
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply scaling
    ('lasso', Lasso())
])
lasso_param_grid = {"lasso__alpha": np.logspace(-8, 2, 100)}
lasso_grid = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)
lasso_base_model = lasso_grid.best_estimator_

# Predict and Evaluate Lasso
lasso_predictions = lasso_base_model.predict(X_test)
lasso_metrics = calculate_regression_metrics(y_test, lasso_predictions)
print("Lasso Metrics:", lasso_metrics)

# Random Forest with Pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling applied for consistency
    ('rf', RandomForestRegressor(random_state=RANDOM_STATE))
])
rf_param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__max_depth": [None, 10, 20]
}
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_mean_squared_error')
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_

# Predict and Evaluate Random Forest
rf_predictions = rf_model.predict(X_test)
rf_metrics = calculate_regression_metrics(y_test, rf_predictions)
print("Random Forest Metrics:", rf_metrics)

# XGBoost with Pipeline
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Apply scaling
    ('xgb', XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'))
])
xgb_param_grid = {
    "xgb__n_estimators": [50, 100, 200],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.1, 0.2],
    "xgb__subsample": [0.6, 0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='neg_mean_squared_error')
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_

# Predict and Evaluate XGBoost
xgb_predictions = xgb_model.predict(X_test)
xgb_metrics = calculate_regression_metrics(y_test, xgb_predictions)
print("XGBoost Metrics:", xgb_metrics)

# ARIMA Model
arima_model = optimize_arima(
    y_train=y_train,
    p_values=[0, 1, 2],
    d_values=[0, 1],
    q_values=[0, 1, 2]
)
arima_predictions = arima_model.predict(
    start=len(y_train) + len(y_val1),
    end=len(y_train) + len(y_val1) + len(y_test) - 1
)
arima_metrics = calculate_regression_metrics(y_test, arima_predictions)
print("ARIMA Metrics:", arima_metrics)

# LSTM Model
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
print("LSTM Metrics:", lstm_metrics)


### Ensemble Models ###

# Ensure consistent preprocessing for all datasets
# Identify common features and align datasets
common_features = X_train.columns.intersection(X_test.columns)

X_train = X_train[common_features]
X_val1 = X_val1[common_features]
X_test = X_test[common_features]

# Define a reusable preprocessing pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Ensure all base models use the preprocessing pipeline
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lasso_base_model)
])
lasso_pipeline.fit(X_train, y_train)
lasso_predictions = lasso_pipeline.predict(X_test)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
rf_pipeline.fit(X_train, y_train)
rf_predictions = rf_pipeline.predict(X_test)

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])
xgb_pipeline.fit(X_train, y_train)
xgb_predictions = xgb_pipeline.predict(X_test)

# Handle ARIMA predictions separately (does not use feature-based input)
arima_predictions = arima_model.predict(
    start=len(y_train) + len(y_val1),
    end=len(y_train) + len(y_val1) + len(y_test) - 1
)

# Reshape and predict for LSTM
lstm_predictions = lstm_model.predict(X_test.values.reshape(-1, 1, X_test.shape[1]))

# Generate predictions for each base model on training, validation, and test sets
ensemble_predictions_train = np.column_stack([
    lasso_pipeline.predict(X_train),
    rf_pipeline.predict(X_train),
    xgb_pipeline.predict(X_train),
    arima_model.predict(start=0, end=len(y_train) - 1),
    lstm_model.predict(X_train.values.reshape(-1, 1, X_train.shape[1])),
])

ensemble_predictions_val = np.column_stack([
    lasso_pipeline.predict(X_val1),
    rf_pipeline.predict(X_val1),
    xgb_pipeline.predict(X_val1),
    arima_model.predict(start=len(X_train), end=len(X_train) + len(y_val1) - 1),
    lstm_model.predict(X_val1.values.reshape(-1, 1, X_val1.shape[1])),
])

ensemble_predictions_test = np.column_stack([
    lasso_predictions,
    rf_predictions,
    xgb_predictions,
    arima_predictions,
    lstm_predictions,
])


### Neural Network Ensemble ###

# Hyperparameter grid for NN Ensemble
nn_param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "num_epochs": [50, 100, 200],
    "hidden_size": [50, 100],  # Example: additional hidden layer size parameter
}

# Hyperparameter optimization for NN Ensemble
best_nn_params = None
best_nn_model = None
best_nn_metrics = None
min_mse = float("inf")

for learning_rate in nn_param_grid["learning_rate"]:
    for num_epochs in nn_param_grid["num_epochs"]:
        for hidden_size in nn_param_grid["hidden_size"]:
            print(f"Tuning NN Ensemble with LR={learning_rate}, Epochs={num_epochs}, Hidden Size={hidden_size}")
            
            # Initialize and train model
            nn_model = NeuralNetworkEnsembleWrapper(
                input_size=ensemble_predictions_train.shape[1],
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                hidden_size=hidden_size,
            )
            nn_model.fit(ensemble_predictions_train, y_train)
            nn_preds = nn_model.predict(ensemble_predictions_test)
            nn_metrics = calculate_regression_metrics(y_test, nn_preds)
            
            # Update best model based on MSE
            if nn_metrics["MSE"] < min_mse:
                min_mse = nn_metrics["MSE"]
                best_nn_params = {"learning_rate": learning_rate, "num_epochs": num_epochs, "hidden_size": hidden_size}
                best_nn_model = nn_model
                best_nn_metrics = nn_metrics

# Display the best NN Ensemble metrics
print(f"Best NN Ensemble Parameters: {best_nn_params}")
print(f"Best NN Ensemble Metrics: {best_nn_metrics}")

# Re-train the best model on the full training set (to use in Kalman Filter predictions)
best_nn_model = NeuralNetworkEnsembleWrapper(
    input_size=ensemble_predictions_train.shape[1],
    learning_rate=best_nn_params["learning_rate"],
    num_epochs=best_nn_params["num_epochs"],
    hidden_size=best_nn_params["hidden_size"],
)

best_nn_model.fit(ensemble_predictions_train, y_train)

# Generate predictions for Kalman Filter
nn_ensemble_predictions = best_nn_model.predict(ensemble_predictions_test)  # Test set predictions
nn_predictions_trimmed = best_nn_model.predict(ensemble_predictions_val)  # Kalman validation predictions

# Generate metrics for Neural Network Ensemble
nn_ensemble_metrics = calculate_regression_metrics(y_test, nn_ensemble_predictions)


### Stacking Ensemble ###
# Define hyperparameter grid for the meta-model (Lasso)
meta_param_grid = {"alpha": np.logspace(-6, 2, 10)}

# Optimize hyperparameters for the meta-model (LassoWrapper with pipeline)
lasso_meta_model_stacking, _ = optimize_model_hyperparameters(
    LassoWrapper,
    meta_param_grid,
    ensemble_predictions_train,  # Base model predictions as input for meta-model
    y_train,
    validation_data=(ensemble_predictions_val, y_val1),
    n_jobs=1,
)

# Define and fit the stacking ensemble with selected base models (excluding ARIMA and LSTM)
stacking_model = StackingRegressorWrapper(
    base_models=[
        ("lasso", lasso_base_model),
        ("rf", rf_model),
        ("xgb", xgb_model),
    ],
    meta_model=lasso_meta_model_stacking,
)

# Fit the stacking model using training and validation sets
stacking_model.fit(
    X_train,  # Features for base models
    y_train,  # Training target
    X_val1,   # Features for meta-model validation
    y_val1,   # Validation target for meta-model
)

# Generate predictions on the test set
stacking_predictions = stacking_model.predict(X_test)

# Evaluate stacking ensemble performance
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)
print("Updated Stacking Ensemble Metrics:", stacking_metrics)





### Updated Weighted Average Ensemble ###
def optimize_weights_sequential(predictions, y_true):
    """
    Optimize weights for a weighted ensemble using MSE as the objective.
    
    Args:
        predictions (np.ndarray): Array of shape (n_models, n_samples) containing predictions from models.
        y_true (np.ndarray): Array of shape (n_samples,) containing true target values.
    
    Returns:
        np.ndarray: Optimized weights for the ensemble.
    """
    def loss_function(weights):
        # Compute weighted ensemble prediction
        ensemble_prediction = np.dot(weights, predictions)
        # Mean squared error as loss
        return np.mean((y_true - ensemble_prediction) ** 2)

    # Initialize equal weights
    initial_weights = np.ones(predictions.shape[0]) / predictions.shape[0]
    
    # Constraints: weights must sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    # Bounds: weights must be between 0 and 1
    bounds = [(0, 1)] * predictions.shape[0]
    
    # Minimize the loss function
    result = minimize(loss_function, initial_weights, constraints=constraints, bounds=bounds)
    
    if not result.success:
        raise ValueError("Weight optimization failed: " + result.message)
    
    return result.x


# Optimize weights for weighted average ensemble
optimized_weights = optimize_weights_sequential(ensemble_predictions_train.T, y_train)
weighted_ensemble_predictions = np.dot(optimized_weights, ensemble_predictions_test.T)
weighted_ensemble_metrics = calculate_regression_metrics(y_test, weighted_ensemble_predictions)
print("Updated Weighted Average Ensemble Metrics:", weighted_ensemble_metrics)

### Updated Lasso Ensemble ###
# # Define hyperparameter grid for Lasso Ensemble
# lasso_param_grid = {"lasso__alpha": np.logspace(-6, 2, 10)}

# # Define the Lasso pipeline
# lasso_pipeline = Pipeline([
#     ("scaler", StandardScaler()),  # StandardScaler avoids data leakage
#     ("lasso", Lasso())
# ])

# # Hyperparameter tuning using GridSearchCV-like optimization
# lasso_ensemble_model, best_lasso_params = optimize_model_hyperparameters(
#     lambda: lasso_pipeline,  # Pass pipeline for optimization
#     lasso_param_grid,
#     ensemble_predictions_train,  # Use base model predictions
#     y_train,
#     validation_data=(ensemble_predictions_val, y_val1),  # Validation split
#     n_jobs=1
# )

# # Fit and evaluate the best Lasso ensemble model
# lasso_ensemble_model.fit(ensemble_predictions_train, y_train)
# lasso_ensemble_predictions = lasso_ensemble_model.predict(ensemble_predictions_test)
# lasso_ensemble_metrics = calculate_regression_metrics(y_test, lasso_ensemble_predictions)

# # Display Lasso Ensemble results
# print(f"Best Lasso Ensemble Parameters: {best_lasso_params}")
# print(f"Updated Lasso Ensemble Metrics: {lasso_ensemble_metrics}")
# Define the Lasso ensemble pipeline with hyperparameter tuning
lasso_param_grid = {"lasso__alpha": np.logspace(-6, 2, 10)}
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),  # StandardScaler avoids data leakage
    ("lasso", Lasso())
])

lasso_ensemble_model, _ = optimize_model_hyperparameters(
    lambda: lasso_pipeline,  # Use pipeline in hyperparameter tuning
    lasso_param_grid,
    ensemble_predictions_train,  # Base model predictions as input
    y_train,
    validation_data=(ensemble_predictions_val, y_val1),
    n_jobs=1
)

# Fit the Lasso ensemble
lasso_ensemble_model.fit(ensemble_predictions_train, y_train)

# Predict and evaluate
lasso_ensemble_predictions = lasso_ensemble_model.predict(ensemble_predictions_test)
lasso_ensemble_metrics = calculate_regression_metrics(y_test, lasso_ensemble_predictions)
print("Updated Lasso Ensemble Metrics:", lasso_ensemble_metrics)


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

# Compute optimized weights for the weighted ensemble
optimized_weights = optimize_weights_sequential(ensemble_predictions_train.T, y_train)

# Generate weighted ensemble predictions on the test set
optimized_weighted_predictions = np.dot(optimized_weights, ensemble_predictions_test.T)

# Trim weighted predictions to match y_test length
weighted_predictions_trimmed = optimized_weighted_predictions[:len(y_test)]

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

# Neural Network Ensemble Integration with Kalman Filters
nn_predictions_trimmed = nn_ensemble_predictions[:len(y_test)]  # Align predictions with y_test

# Constant Velocity Kalman Filter (CVKF) for Neural Network Ensemble
cvkf_params_nn, cvkf_preds_nn = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    nn_predictions_trimmed,
    y_test,
    n_jobs=1
)
cvkf_metrics_nn = calculate_regression_metrics(y_test, cvkf_preds_nn)

# Financial Model Kalman Filter (FMKF) for Neural Network Ensemble
fmkf_params_nn, fmkf_preds_nn = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    nn_predictions_trimmed,
    y_test,
    n_jobs=1
)
fmkf_metrics_nn = calculate_regression_metrics(y_test, fmkf_preds_nn)
print("NN Ensemble + CVKF Metrics:", cvkf_metrics_nn)
print("NN Ensemble + FMKF Metrics:", fmkf_metrics_nn)

### Summary ###
all_metrics = {
    "T-1 Baseline": t_minus_1_metrics,
    f"Windowed Average (Window={WINDOW_SIZE})": windowed_avg_metrics,
    "Lasso (Base)": calculate_regression_metrics(y_test, lasso_predictions),
    "Random Forest": calculate_regression_metrics(y_test, rf_predictions),
    "XGBoost": calculate_regression_metrics(y_test, xgb_predictions),
    "Stacking Ensemble": stacking_metrics,
    "Weighted Ensemble": calculate_regression_metrics(y_test, optimized_weighted_predictions),
    "Lasso Ensemble": lasso_ensemble_metrics,
    "NN Ensemble": nn_ensemble_metrics,
    "CVKF (Stacking)": cvkf_metrics_stack,
    "FMKF (Stacking)": fmkf_metrics_stack,
    "CVKF (Weighted)": cvkf_metrics_weight,
    "FMKF (Weighted)": fmkf_metrics_weight,
    "CVKF (Lasso)": cvkf_metrics_lasso,
    "FMKF (Lasso)": fmkf_metrics_lasso,
    "CVKF (NN)": cvkf_metrics_nn,
    "FMKF (NN)": fmkf_metrics_nn,
}

metrics_df = pd.DataFrame(all_metrics).T
print("Final Model Metrics:\n", metrics_df)





