import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from metrics.metrics import (
    calculate_regression_metrics, calculate_classification_metrics,
    calculate_regression_information_gain, calculate_classification_information_gain
)
from models.models import (
    LinearRegressionWrapper, LassoWrapper, LogisticRegressionWrapper, StackingRegressorWrapper,
    ensemble_weighted_average, optimize_model_hyperparameters
)
from kalman_filter.kalman_filter import EnsembleKalmanFilter
from preprocessing.preprocessing import (
    normalize_data, create_lagged_features, split_temporal_data
)

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
X, y = preprocess_data(
    file_path="simulated_series_cont.csv", 
    target_column="return", 
    lag_features={"return": 1}
)

# Train-validation-test split for time-series data
def train_validation_test_split(X, y, val_size=0.2, test_size=0.2):
    total_len = len(X)
    test_len = int(total_len * test_size)
    val_len = int(total_len * val_size)

    train_idx = range(0, total_len - val_len - test_len)
    val_idx = range(total_len - val_len - test_len, total_len - test_len)
    test_idx = range(total_len - test_len, total_len)

    return (
        X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    )

X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y)

# Step 2: Train and Optimize Ensemble Models for Regression
param_grid = {"alpha": np.logspace(-3, 1, 10)}
lasso_model, _ = optimize_model_hyperparameters(
    LassoWrapper, param_grid, X_train, y_train, validation_data=(X_val, y_val)
)

# Stacking Ensemble
base_models = [
    ("linear", LinearRegressionWrapper()),
    ("lasso", lasso_model)
]
meta_model = LassoWrapper(alpha=0.1)
stacking_model = StackingRegressorWrapper(base_models, meta_model)
stacking_model.fit(X_train, y_train)

# Generate predictions for test set
lasso_predictions = lasso_model.predict(X_test)
stacking_predictions = stacking_model.predict(X_test)

# Weighted Average Ensemble
ensemble_predictions = np.array([lasso_predictions, stacking_predictions])
weights = np.ones(ensemble_predictions.shape[0]) / ensemble_predictions.shape[0]  # Initial weights
weighted_predictions = ensemble_weighted_average(ensemble_predictions, weights)

# Step 4: Kalman Filter Integration for Regression
F = np.eye(1)
H = np.eye(1)
Q = np.eye(1) * 0.1
R = np.eye(1) * 0.1
P = np.eye(1)
x = np.array([0])

kalman_filter = EnsembleKalmanFilter(F, H, Q, R, P, x)
kalman_predictions = []
for pred in stacking_predictions:
    kalman_filter.predict()
    kalman_filter.update(np.array([pred]))
    kalman_predictions.append(kalman_filter.x[0])

# Step 5: Compute Results Post-Kalman Filter for Regression
pre_kalman_metrics = calculate_regression_metrics(y_test, stacking_predictions)
post_kalman_metrics = calculate_regression_metrics(y_test, kalman_predictions)

# Step 6: Compute Information Gain for Regression
pre_kalman_info_gain = calculate_regression_information_gain(y_test, stacking_predictions)
post_kalman_info_gain = calculate_regression_information_gain(y_test, kalman_predictions)

# Output Regression Metrics
print("Regression Pre-Kalman Filter Metrics:", pre_kalman_metrics)
print("Regression Post-Kalman Filter Metrics:", post_kalman_metrics)
print("Regression Pre-Kalman Filter Information Gain:", pre_kalman_info_gain)
print("Regression Post-Kalman Filter Information Gain:", post_kalman_info_gain)

# Step 7: Classification Workflow with Preprocessing
X_class, y_class = preprocess_data(
    file_path="simulated_series_bin.csv", 
    target_column="binary", 
    lag_features={"binary": 1}
)

# Binarize the target variable to ensure discrete classes
y_class = (y_class > 0).astype(int)  # Example: 1 for positive values, 0 for non-positive values

# Train-validation-test split
X_train_class, X_val_class, X_test_class, y_train_class, y_val_class, y_test_class = train_validation_test_split(
    X_class, y_class
)

# Train and Optimize Classification Models
param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
logistic_model, best_params = optimize_model_hyperparameters(
    LogisticRegressionWrapper, param_grid,
    X_train_class, y_train_class, validation_data=(X_val_class, y_val_class),
    scoring="accuracy"
)

print("Best Parameters:", best_params)

# Generate Predictions for Classification
logistic_predictions = logistic_model.predict(X_test_class)
logistic_pred_proba = logistic_model.predict_proba(X_test_class)

# Weighted Average Ensemble for Classification
classification_ensemble_predictions = np.array([logistic_predictions])
classification_weights = np.ones(classification_ensemble_predictions.shape[0]) / classification_ensemble_predictions.shape[0]
classification_weighted_predictions = ensemble_weighted_average(classification_ensemble_predictions, classification_weights)

# Kalman Filter Integration for Classification
kalman_class_filter = EnsembleKalmanFilter(F, H, Q, R, P, x)
kalman_class_predictions = []
for pred in classification_weighted_predictions:
    kalman_class_filter.predict()
    kalman_class_filter.update(np.array([pred]))
    kalman_class_predictions.append(kalman_class_filter.x[0])

# Compute Metrics and Information Gain for Classification
classification_metrics = calculate_classification_metrics(y_test_class, classification_weighted_predictions)
classification_info_gain = calculate_classification_information_gain(y_test_class, logistic_pred_proba)

# Output Classification Metrics
print("Classification Metrics:", classification_metrics)
print("Classification Information Gain:", classification_info_gain)

