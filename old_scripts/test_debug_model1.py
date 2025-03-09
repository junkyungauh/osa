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
import matplotlib.pyplot as plt

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

# Debugging: Ensure splits are valid and do not overlap
print("Train range:", X_train.index.min(), "-", X_train.index.max())
print("Validation range:", X_val.index.min(), "-", X_val.index.max())
print("Test range:", X_test.index.min(), "-", X_test.index.max())

# Debugging: Inspect the correlation between features and target
print("Feature-Target Correlations:")
print(X_train.corrwith(y_train))

# Debugging: Check the distribution of the target variable
plt.hist(y_train, bins=50, alpha=0.7, label="Train")
plt.hist(y_val, bins=50, alpha=0.7, label="Validation")
plt.hist(y_test, bins=50, alpha=0.7, label="Test")
plt.legend()
plt.title("Target Variable Distribution")
plt.show()

# Step 2: Train and Optimize Ensemble Models for Regression
# Set alpha to 10^-2 as determined from the validation curve
lasso_model = LassoWrapper(alpha=10**-2)
lasso_model.fit(X_train, y_train)

# Debugging: Inspect Lasso coefficients
print("Lasso Coefficients:", lasso_model.coef_)
print("Non-zero Coefficients:", np.sum(lasso_model.coef_ != 0))

# Stacking Ensemble
base_models = [
    ("linear", LinearRegressionWrapper()),
    ("lasso", lasso_model)
]
meta_model = LinearRegressionWrapper()  # Simplified meta-model for potential underfitting
stacking_model = StackingRegressorWrapper(base_models, meta_model)
stacking_model.fit(X_train, y_train)

# Generate predictions for test set
lasso_predictions = lasso_model.predict(X_test)
stacking_predictions = stacking_model.predict(X_test)

# Debugging: Compare residuals for stacking predictions
residuals = y_test - stacking_predictions
plt.hist(residuals, bins=50, alpha=0.7)
plt.title("Residuals Histogram")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# Debugging: Compare baseline predictions (return_lag1) vs. stacking predictions
y_baseline = X_test["return_lag1"]
plt.plot(y_test.values, label="True")
plt.plot(y_baseline.values, label="Baseline")
plt.plot(stacking_predictions, label="Stacking Predictions")
plt.legend()
plt.title("Comparison of Baseline and Stacking Predictions")
plt.show()

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

# Debugging: Print metrics to compare pre- and post-Kalman results
print("Regression Pre-Kalman Filter Metrics:", pre_kalman_metrics)
print("Regression Post-Kalman Filter Metrics:", post_kalman_metrics)

# Output Validation for Classification
X_class, y_class = preprocess_data(
    file_path="simulated_series_bin.csv", 
    target_column="binary", 
    lag_features={"binary": 1}
)

# Debugging: Confirm binary target variable is correctly binarized
y_class = (y_class > 0).astype(int)
print("Unique values in binary target:", y_class.unique())

# Train-validation-test split
X_train_class, X_val_class, X_test_class, y_train_class, y_val_class, y_test_class = train_validation_test_split(
    X_class, y_class
)

# Debugging: Print sample sizes of splits
print(f"Classification Train size: {len(X_train_class)}")
print(f"Classification Validation size: {len(X_val_class)}")
print(f"Classification Test size: {len(X_test_class)}")
