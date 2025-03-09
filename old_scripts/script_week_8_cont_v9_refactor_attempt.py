import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim

# Import custom utilities and modules
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

# Constants
RANDOM_STATE = 42
WINDOW_SIZE = 10

# Hyperparameter Grids
LASSO_PARAM_GRID = {"alpha": np.logspace(-8, 2, 100)}
RF_PARAM_GRID = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0]
}
STACKING_META_PARAM_GRID = {"alpha": np.logspace(-8, 2, 20)}
CVKF_PARAM_GRID = [{"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r]} for q in [0.01, 0.1, 1.0, 10.0] for r in [0.01, 0.1, 1.0, 10.0]]
FMKF_PARAM_GRID = [{"initial_state": np.array([0.0]), "Q_diag": [q], "R_diag": [r], "alpha": [a], "beta": [b]} for q in [0.01, 0.1, 1.0, 10.0] for r in [0.01, 0.1, 1.0, 10.0] for a in [0.4, 0.6, 0.8, 1.0] for b in [0.05, 0.1, 0.2, 0.4]]


# File paths
data_path = 'simulated_series_cont.csv'

def preprocess_and_split(data_path):
    X, y = preprocess_data_with_general_features(
        file_path=data_path,
        target_column='return',
        lag_steps=[1, 2, 3],
        rolling_window=10,
        ema_window=5
    )
    return five_way_split(
        X, y, train_size=0.5, val1_size=0.15, val2_size=0.1, kalman_size=0.1, test_size=0.15
    )

X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = preprocess_and_split(data_path)


# Base Models
def train_base_models(X_train, y_train, X_test, y_test):
    # Lasso
    lasso_pipeline = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())])
    lasso_grid = GridSearchCV(lasso_pipeline, LASSO_PARAM_GRID, cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_train, y_train)
    lasso_model = lasso_grid.best_estimator_
    lasso_metrics = calculate_regression_metrics(y_test, lasso_model.predict(X_test))

    # Random Forest
    rf_pipeline = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(random_state=RANDOM_STATE))])
    rf_grid = GridSearchCV(rf_pipeline, RF_PARAM_GRID, cv=5, scoring='neg_mean_squared_error')
    rf_grid.fit(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    rf_metrics = calculate_regression_metrics(y_test, rf_model.predict(X_test))

    # XGBoost
    xgb_pipeline = Pipeline([("scaler", StandardScaler()), ("xgb", XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'))])
    xgb_grid = GridSearchCV(xgb_pipeline, XGB_PARAM_GRID, cv=5, scoring='neg_mean_squared_error')
    xgb_grid.fit(X_train, y_train)
    xgb_model = xgb_grid.best_estimator_
    xgb_metrics = calculate_regression_metrics(y_test, xgb_model.predict(X_test))

    return lasso_model, rf_model, xgb_model, lasso_metrics, rf_metrics, xgb_metrics


# Neural Network Ensemble
def optimize_nn_ensemble(X_train, y_train, X_test, y_test):
    nn_param_grid = {"learning_rate": [0.001, 0.01, 0.1], "num_epochs": [50, 100, 200], "hidden_size": [50, 100]}
    best_model, best_params, best_metrics = None, None, float("inf")
    for lr in nn_param_grid["learning_rate"]:
        for epochs in nn_param_grid["num_epochs"]:
            for size in nn_param_grid["hidden_size"]:
                model = NeuralNetworkEnsembleWrapper(input_size=X_train.shape[1], learning_rate=lr, num_epochs=epochs, hidden_size=size)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                metrics = calculate_regression_metrics(y_test, preds)
                if metrics["MSE"] < best_metrics:
                    best_model, best_params, best_metrics = model, {"lr": lr, "epochs": epochs, "size": size}, metrics
    return best_model, best_params, best_metrics


