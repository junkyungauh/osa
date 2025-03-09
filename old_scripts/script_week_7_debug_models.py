import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics
from models.models import (
    LinearRegressionWrapper, LassoWrapper, LogisticRegressionWrapper, StackingRegressorWrapper,
    ensemble_weighted_average, optimize_model_hyperparameters
)
from preprocessing.preprocessing import normalize_data, create_lagged_features, split_temporal_data

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

# Step 2: Base Model Hyperparameter Tuning
# Lasso Hyperparameter Tuning
param_grid = {"alpha": np.logspace(-6, 2, 50)}
lasso_model, _ = optimize_model_hyperparameters(
    LassoWrapper, param_grid, X_train, y_train, validation_data=(X_val, y_val)
)

# Debugging: Validation Curve for Lasso
alphas = np.logspace(-6, 2, 50)
train_scores, val_scores = validation_curve(
    LassoWrapper(), X_train, y_train, param_name="alpha", param_range=alphas, scoring="neg_mean_squared_error"
)
plt.plot(alphas, np.mean(-train_scores, axis=1), label="Train")
plt.plot(alphas, np.mean(-val_scores, axis=1), label="Validation")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.legend()
plt.title("Validation Curve for Lasso Alpha")
plt.show()

# Linear Regression Hyperparameter Tuning
linear_reg_model = LinearRegressionWrapper()
linear_reg_model.fit(X_train, y_train)

# Debugging: Inspect Lasso coefficients
print("Lasso Coefficients:", lasso_model.model.coef_)
print("Non-zero Coefficients:", np.sum(lasso_model.model.coef_ != 0))

# Random Forest Hyperparameter Tuning
rf_param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
random_forest_model, _ = optimize_model_hyperparameters(
    RandomForestRegressor(), rf_param_grid, X_train, y_train, validation_data=(X_val, y_val)
)

# Step 3: Ensemble Method Hyperparameter Tuning
# Stacking Ensemble
base_models = [
    ("linear", linear_reg_model),
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

# Debugging: Evaluate weighted ensemble predictions
ensemble_metrics = calculate_regression_metrics(y_test, weighted_predictions)
print("Weighted Ensemble Metrics:", ensemble_metrics)

# Output Results
lasso_metrics = calculate_regression_metrics(y_test, lasso_predictions)
stacking_metrics = calculate_regression_metrics(y_test, stacking_predictions)

print("Lasso Regression Metrics:", lasso_metrics)
print("Stacking Regression Metrics:", stacking_metrics)
print("Weighted Ensemble Metrics:", ensemble_metrics)
