import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from models.models import LassoWrapper, optimize_model_hyperparameters
from preprocessing.preprocessing import normalize_data

# Path to your data
cont_data_path = 'simulated_series_cont.csv'

# Constants
LASSO_PARAM_GRID = {"alpha": np.logspace(-8, 2, 50)}
WINDOW_SIZE = 10
RANDOM_STATE = 42

# Utility functions for metrics
def calculate_regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }

# Preprocessing with general-purpose features
def preprocess_data_with_general_features(file_path, target_column, lag_steps=None, rolling_window=None, ema_window=None, normalize=True):
    data = pd.read_csv(file_path)
    feature_data = pd.DataFrame(index=data.index)

    # Lagged Features
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)

    # Rolling Features
    if rolling_window:
        feature_data[f'{target_column}_roll_mean_{rolling_window}'] = data[target_column].rolling(window=rolling_window, closed='right').mean()
        feature_data[f'{target_column}_roll_std_{rolling_window}'] = data[target_column].rolling(window=rolling_window, closed='right').std()

    # Exponential Moving Average
    if ema_window:
        feature_data[f'{target_column}_ema_{ema_window}'] = data[target_column].ewm(span=ema_window, adjust=False).mean()

    # First Differences
    feature_data[f'{target_column}_diff1'] = data[target_column].diff()

    # Drop rows with NaNs introduced by feature engineering
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]

    # Normalize features if specified
    X = normalize_data(feature_data) if normalize else feature_data
    y = data.loc[feature_data.index, target_column]
    return X, y

# Load and preprocess data
X_cont, y_cont = preprocess_data_with_general_features(
    file_path=cont_data_path,
    target_column="return",
    lag_steps=[1, 2, 3],
    rolling_window=10,
    ema_window=5,
    normalize=True
)

# Debugging: Random features and random targets
X_random = pd.DataFrame(np.random.randn(*X_cont.shape), columns=X_cont.columns)
y_random = pd.Series(np.random.randn(len(y_cont)), name="random_target")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_random, y_random, test_size=0.2, random_state=RANDOM_STATE, shuffle=False)

# Train Lasso on random features
lasso_random, _ = optimize_model_hyperparameters(LassoWrapper, LASSO_PARAM_GRID, X_train, y_train, validation_data=(X_test, y_test))
lasso_random_preds = lasso_random.predict(X_test)
random_metrics = calculate_regression_metrics(y_test, lasso_random_preds)
print("Lasso Metrics with Random Features:", random_metrics)

# Permutation Test for Target
permuted_mae = []
for _ in range(50):
    y_perm = shuffle(y_cont, random_state=RANDOM_STATE)
    lasso_perm, _ = optimize_model_hyperparameters(LassoWrapper, LASSO_PARAM_GRID, X_train, y_perm[:len(X_train)], validation_data=(X_test, y_perm[len(X_train):]))
    permuted_preds = lasso_perm.predict(X_test)
    permuted_mae.append(mean_absolute_error(y_perm[len(X_train):], permuted_preds))
print("Mean Permuted MAE:", np.mean(permuted_mae))

# Random Noise in Target
noise_std = y_cont.std() * 0.5
y_noisy = y_cont + np.random.normal(0, noise_std, len(y_cont))
lasso_noisy, _ = optimize_model_hyperparameters(LassoWrapper, LASSO_PARAM_GRID, X_train, y_noisy[:len(X_train)], validation_data=(X_test, y_noisy[len(X_train):]))
lasso_noisy_preds = lasso_noisy.predict(X_test)
noisy_metrics = calculate_regression_metrics(y_noisy[len(X_train):], lasso_noisy_preds)
print("Lasso Metrics with Noisy Target:", noisy_metrics)

# Correlation Check
correlations = X_random.corrwith(y_cont)
print("Correlations between random features and target:", correlations)

# Cross-Validation on Minimal Features
cv_mae = cross_val_score(LassoWrapper(alpha=0.1), X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
print("Cross-Validation MAE on Random Features:", -np.mean(cv_mae))

# Residual Analysis
best_model_preds = lasso_random.predict(X_test)
residuals = y_test - best_model_preds

plt.scatter(best_model_preds, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predictions")
plt.show()
