import numpy as np
import pandas as pd
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingClassifier
from metrics.metrics import (
    calculate_regression_metrics, calculate_classification_metrics
)
from models.models import (
    LinearRegressionWrapper, LassoWrapper, LogisticRegressionWrapper, StackingRegressorWrapper,
    ensemble_weighted_average, optimize_model_hyperparameters
)
from kalman_filter.kalman_filter import (
    ConstantVelocityKalmanFilter, FinancialModelKalmanFilter, optimize_kalman_hyperparameters
)
from preprocessing.preprocessing import normalize_data, create_lagged_features

# Utility: Four-way split for training, ensemble validation, Kalman filter validation, and testing
def four_way_split(data, val1_size=0.2, val2_size=0.2, test_size=0.2):
    total_len = len(data)
    test_len = int(total_len * test_size)
    val2_len = int(total_len * val2_size)
    val1_len = int(total_len * val1_size)

    train_idx = range(0, total_len - val1_len - val2_len - test_len)
    val1_idx = range(total_len - val1_len - val2_len - test_len, total_len - val2_len - test_len)
    val2_idx = range(total_len - val2_len - test_len, total_len - test_len)
    test_idx = range(total_len - test_len, total_len)

    return (
        data.iloc[train_idx], data.iloc[val1_idx], data.iloc[val2_idx], data.iloc[test_idx]
    )

def preprocess_data(file_path, target_column, lag_features=None, normalize=True, binary_threshold=None):
    """Load and preprocess the data."""
    data = pd.read_csv(file_path)
    
    # Create lagged features
    if lag_features:
        for column, lag in lag_features.items():
            data = create_lagged_features(data, column, lag)
    
    # Normalize the data
    if normalize:
        data = normalize_data(data)
    
    # Convert continuous labels to binary if threshold is provided
    if binary_threshold is not None:
        data[target_column] = (data[target_column] > binary_threshold).astype(int)
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

# Load datasets
X_cont, y_cont = preprocess_data(
    file_path="simulated_series_cont.csv", 
    target_column="return", 
    lag_features={"return": 1}
)
X_bin, y_bin = preprocess_data(
    file_path="simulated_series_bin.csv", 
    target_column="binary", 
    lag_features={"binary": 1},
    binary_threshold=0.5  # Adjust threshold based on your data
)

# Four-way split for continuous and discrete tasks
X_train_cont, X_val1_cont, X_val2_cont, X_test_cont = four_way_split(X_cont)
y_train_cont, y_val1_cont, y_val2_cont, y_test_cont = four_way_split(y_cont)

X_train_bin, X_val1_bin, X_val2_bin, X_test_bin = four_way_split(X_bin)
y_train_bin, y_val1_bin, y_val2_bin, y_test_bin = four_way_split(y_bin)

# Step 1: Train Ensemble Models (Baseline)
# Continuous task
param_grid_cont = {"alpha": np.logspace(-3, 1, 10)}
lasso_model_cont, _ = optimize_model_hyperparameters(
    LassoWrapper, param_grid_cont, X_train_cont, y_train_cont, validation_data=(X_val1_cont, y_val1_cont)
)
base_models_cont = [("linear", LinearRegressionWrapper()), ("lasso", lasso_model_cont)]
meta_model_cont = LassoWrapper(alpha=0.1)
stacking_model_cont = StackingRegressorWrapper(base_models_cont, meta_model_cont)
stacking_model_cont.fit(X_train_cont, y_train_cont)

# Discrete task
param_grid_bin = {"C": [0.1, 1, 10], "solver": ["lbfgs"]}
logistic_model_bin, _ = optimize_model_hyperparameters(
    LogisticRegressionWrapper, param_grid_bin, X_train_bin, y_train_bin, validation_data=(X_val1_bin, y_val1_bin)
)
base_models_bin = [("logistic", logistic_model_bin)]
meta_model_bin = LogisticRegressionWrapper(C=1.0, solver="lbfgs")
stacking_model_bin = StackingClassifier(
    estimators=base_models_bin, final_estimator=meta_model_bin
)
stacking_model_bin.fit(X_train_bin, y_train_bin)

# Step 2: Generate Predictions for Kalman Filters
stacking_preds_cont = stacking_model_cont.predict(X_val2_cont)
stacking_preds_bin = stacking_model_bin.predict_proba(X_val2_bin)[:, 1]  # Use probabilities for Kalman filters

# Step 3: Optimize Kalman Filters
constant_velocity_grid = [
    {"initial_state": [np.array([stacking_preds_cont[0], 0])], "Q_diag": [q], "R_diag": [r]}
    for q in [0.01, 0.1, 1.0]
    for r in [0.01, 0.1, 1.0]
]

financial_model_grid = [
    {"initial_state": [np.array([stacking_preds_cont[0]])], "Q_diag": [q], "R_diag": [r], "alpha": [a], "beta": [b]}
    for q in [0.01, 0.1, 1.0]
    for r in [0.01, 0.1, 1.0]
    for a in [0.6, 0.8, 1.0]
    for b in [0.1, 0.2, 0.4]
]

# Optimize Constant Velocity Kalman Filter
cvkf_params_cont, cvkf_preds_cont = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params), constant_velocity_grid, stacking_preds_cont, y_val2_cont
)
cvkf_params_bin, cvkf_preds_bin = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params), constant_velocity_grid, stacking_preds_bin, y_val2_bin
)

# Optimize Financial Model Kalman Filter
fmkf_params_cont, fmkf_preds_cont = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params), financial_model_grid, stacking_preds_cont, y_val2_cont
)
fmkf_params_bin, fmkf_preds_bin = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params), financial_model_grid, stacking_preds_bin, y_val2_bin
)

# Step 4: Final Evaluation on Test Set
# Continuous task
test_stacking_preds_cont = stacking_model_cont.predict(X_test_cont)
pre_kalman_metrics_cont = calculate_regression_metrics(y_test_cont, test_stacking_preds_cont)
cvkf_metrics_cont = calculate_regression_metrics(y_test_cont, cvkf_preds_cont)
fmkf_metrics_cont = calculate_regression_metrics(y_test_cont, fmkf_preds_cont)

# Discrete task
test_stacking_preds_bin = stacking_model_bin.predict(X_test_bin)
pre_kalman_metrics_bin = calculate_classification_metrics(y_test_bin, test_stacking_preds_bin)

# Binarize continuous Kalman predictions for classification metrics
threshold = 0.5
cvkf_preds_bin_binarized = (np.array(cvkf_preds_bin) >= threshold).astype(int)
fmkf_preds_bin_binarized = (np.array(fmkf_preds_bin) >= threshold).astype(int)

cvkf_metrics_bin = calculate_classification_metrics(y_test_bin, cvkf_preds_bin_binarized)
fmkf_metrics_bin = calculate_classification_metrics(y_test_bin, fmkf_preds_bin_binarized)

# Output Results
print("Continuous Task Metrics:")
print("Pre-Kalman:", pre_kalman_metrics_cont)
print("Constant Velocity Kalman:", cvkf_metrics_cont)
print("Financial Model Kalman:", fmkf_metrics_cont)

print("Discrete Task Metrics:")
print("Pre-Kalman:", pre_kalman_metrics_bin)
print("Constant Velocity Kalman:", cvkf_metrics_bin)
print("Financial Model Kalman:", fmkf_metrics_bin)
