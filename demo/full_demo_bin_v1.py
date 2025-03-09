import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import random
from sklearn.naive_bayes import GaussianNB
from scipy.optimize import minimize
from kalman_filter.kalman_filter import (
    ConstantVelocityKalmanFilter, FinancialModelKalmanFilter, optimize_kalman_hyperparameters
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pywt  # Ensure you have pywavelets installed for wavelet transforms
# from sklearn.metrics import mean_squared_error, accuracy_score
# from sklearn.model_selection import ParameterGrid
# from joblib import Parallel, delayed


# -----------------
# Hyperparameter Configurations
# -----------------
RANDOM_STATE = 42
WINDOW_SIZE = 10

LASSO_PARAM_GRID = {"logisticregression__C": np.logspace(-3, 2, 10)}
RF_PARAM_GRID = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0]
}
NN_PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "learning_rate": [0.001, 0.01],
    "num_epochs": [50, 100]
}
LSTM_PARAM_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "learning_rate": [0.001, 0.01],
    "num_epochs": [50, 100]
}

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


# -----------------
# Utility Functions
# -----------------


def five_way_split(X, y, train_size=0.5, val1_size=0.15, val2_size=0.1, kalman_size=0.1, test_size=0.15):
    """Split data into five subsets."""
    total_len = len(X)
    
    train_len = round(total_len * train_size)
    val1_len = round(total_len * val1_size)
    val2_len = round(total_len * val2_size)
    kalman_len = round(total_len * kalman_size)
    test_len = total_len - train_len - val1_len - val2_len - kalman_len
    
    train_idx = range(0, train_len)
    val1_idx = range(train_len, train_len + val1_len)
    val2_idx = range(train_len + val1_len, train_len + val1_len + val2_len)
    kalman_idx = range(train_len + val1_len + val2_len, train_len + val1_len + val2_len + kalman_len)
    test_idx = range(train_len + val1_len + val2_len + kalman_len, total_len)
    
    return (
        X.iloc[train_idx], X.iloc[val1_idx], X.iloc[val2_idx], X.iloc[kalman_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[val1_idx], y.iloc[val2_idx], y.iloc[kalman_idx], y.iloc[test_idx]
    )


def optimize_model_hyperparameters(model_fn, param_grid, X_train, y_train, validation_data, n_jobs=1):
    """
    Performs hyperparameter optimization using GridSearchCV.

    Args:
        model_fn: A callable that returns an instance of the model.
        param_grid: Dictionary of hyperparameters to search.
        X_train: Training features.
        y_train: Training labels.
        validation_data: Tuple (X_val, y_val) for validation.
        n_jobs: Number of parallel jobs for GridSearchCV.

    Returns:
        best_model: The best model after GridSearchCV.
        best_params: The best parameters from the search.
    """
    model = model_fn()
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=n_jobs,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate classification metrics including Accuracy, Precision, Recall, F1, and AUC.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_pred_proba (array-like, optional): Predicted probabilities for the positive class.

    Returns:
        dict: Dictionary of calculated metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed


def preprocess_data_with_advanced_features(file_path, target_column, lag_steps=None, rolling_window=10):
    """
    Preprocess data for time series modeling with advanced feature engineering.
    Ensures no data leakage by strictly using past and current data for feature generation.

    Args:
        file_path (str): Path to the input CSV file.
        target_column (str): Target column name.
        lag_steps (list): List of lag steps for feature engineering.
        rolling_window (int): Window size for rolling features.

    Returns:
        tuple: Feature DataFrame (X) and target series (y).
    """
    # Load data and parse dates
    data = pd.read_csv(file_path, index_col=0)
    data.index = pd.to_datetime(data.index, errors='coerce')  # Ensure index is datetime
    assert data.index.is_monotonic_increasing, "Dataset is not sorted by time."

    # Fill missing values in the target column
    data[target_column] = data[target_column].interpolate(method='linear').bfill()

    # Initialize feature storage
    features = []
    indices = []

    for end_idx in range(rolling_window, len(data)):
        # Define the current window
        window = data.iloc[end_idx - rolling_window:end_idx]

        # Compute features for the current timestamp
        current_features = {}

        # Rolling statistics
        signal_cols = [col for col in data.columns if col.startswith('sig')]
        for col in signal_cols:
            current_features[f'{col}_roll_mean'] = window[col].mean()
            current_features[f'{col}_roll_std'] = window[col].std()

        # Lagged features
        if lag_steps:
            for lag in lag_steps:
                if end_idx - lag >= 0:
                    current_features[f'{target_column}_lag{lag}'] = data[target_column].iloc[end_idx - lag]

        # Fourier Transform Features
        for col in signal_cols:
            fourier_transform = np.abs(np.fft.fft(window[col].fillna(0)))
            current_features[f'{col}_fft_max'] = np.max(fourier_transform)
            current_features[f'{col}_fft_mean'] = np.mean(fourier_transform)

        # Wavelet Transform Features
        for col in signal_cols:
            coeffs = pywt.wavedec(window[col].fillna(0), 'db1', level=3)
            current_features[f'{col}_wavelet_approx'] = coeffs[0].mean()
            current_features[f'{col}_wavelet_detail1'] = coeffs[1].mean()
            current_features[f'{col}_wavelet_detail2'] = coeffs[2].mean()

        # Add features and corresponding index
        features.append(current_features)
        indices.append(data.index[end_idx])

    # Convert features to DataFrame
    feature_df = pd.DataFrame(features, index=indices)

    # Align target values
    y = data.loc[feature_df.index, target_column]

    return feature_df, y


# -----------------
# Load and Preprocess Data
# -----------------

# Example Usage
# Load and preprocess data with advanced features
X, y = preprocess_data_with_advanced_features(
    file_path='simulated_series_bin.csv',
    target_column='binary',
    lag_steps=[1, 2, 3],
    rolling_window=10
)

# Perform five-way split
X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(
    X, y, train_size=0.5, val1_size=0.15, val2_size=0.05, kalman_size=0.1, test_size=0.2
)

# -----------------
# Baselines
# -----------------

# T-1 Baseline
y_t1_baseline = X_test["binary_lag1"].astype(int)
t1_metrics = {
    "Accuracy": accuracy_score(y_test, y_t1_baseline),
    "Precision": precision_score(y_test, y_t1_baseline, zero_division=0),
    "Recall": recall_score(y_test, y_t1_baseline, zero_division=0),
    "F1": f1_score(y_test, y_t1_baseline, zero_division=0),
    "AUC": roc_auc_score(y_test, y_t1_baseline)
}
print("T-1 Baseline Metrics:", t1_metrics)

# Random Classifier Baseline
def random_classifier(y_true, seed=42):
    random.seed(seed)
    return pd.Series([random.choice([0, 1]) for _ in range(len(y_true))], index=y_true.index)

y_random = random_classifier(y_test)
random_metrics = {
    "Accuracy": accuracy_score(y_test, y_random),
    "Precision": precision_score(y_test, y_random, zero_division=0),
    "Recall": recall_score(y_test, y_random, zero_division=0),
    "F1": f1_score(y_test, y_random, zero_division=0),
    "AUC": roc_auc_score(y_test, y_random)
}
print("Random Classifier Metrics:", random_metrics)

# Rolling Naive Bayes Baseline
def rolling_naive_bayes(train_series, test_series, window_size):
    predictions = []
    rolling_buffer = train_series.tail(window_size)
    
    for test_point in test_series:
        # Fit Naive Bayes on the rolling buffer
        X_train = np.arange(len(rolling_buffer)).reshape(-1, 1)  # Sequential indices as features
        y_train = rolling_buffer.values  # Targets
        
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Predict the test point
        X_test = np.array([[len(rolling_buffer)]]).reshape(-1, 1)
        prediction = model.predict(X_test)
        predictions.append(prediction[0])
        
        # Update rolling buffer
        rolling_buffer = pd.concat([rolling_buffer, pd.Series([test_point])], ignore_index=True)
        if len(rolling_buffer) > window_size:
            rolling_buffer = rolling_buffer.iloc[1:]
    
    return pd.Series(predictions, index=test_series.index)


y_rolling_nb = rolling_naive_bayes(pd.concat([y_train, y_val1, y_val2]), y_test, WINDOW_SIZE)
rolling_nb_metrics = {
    "Accuracy": accuracy_score(y_test, y_rolling_nb),
    "Precision": precision_score(y_test, y_rolling_nb, zero_division=0),
    "Recall": recall_score(y_test, y_rolling_nb, zero_division=0),
    "F1": f1_score(y_test, y_rolling_nb, zero_division=0),
    "AUC": roc_auc_score(y_test, y_rolling_nb)
}
print("Rolling Naive Bayes Metrics:", rolling_nb_metrics)

# -----------------
# Base Models
# -----------------
# -----------------
# Logistic Regression
# -----------------
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logisticregression', LogisticRegression(class_weight='balanced'))  # Handle class imbalance
])
log_reg_grid = GridSearchCV(log_reg_pipeline, LASSO_PARAM_GRID, cv=5, scoring='roc_auc')
log_reg_grid.fit(X_train, y_train)
log_reg_model = log_reg_grid.best_estimator_

log_reg_preds = log_reg_model.predict(X_test)
log_reg_metrics = {
    "Accuracy": accuracy_score(y_test, log_reg_preds),
    "Precision": precision_score(y_test, log_reg_preds, zero_division=0),  # Avoid warning
    "Recall": recall_score(y_test, log_reg_preds, zero_division=0),
    "F1": f1_score(y_test, log_reg_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, log_reg_model.predict_proba(X_test)[:, 1])
}
print("Logistic Regression Metrics:", log_reg_metrics)

# -----------------
# Random Forest
# -----------------
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('randomforest', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))  # Handle class imbalance
])
rf_param_grid = {
    "randomforest__n_estimators": [50, 100, 200],  # Prefixed by 'randomforest__'
    "randomforest__max_depth": [None, 10, 20]
}
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='roc_auc')
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_

rf_preds = rf_model.predict(X_test)
rf_metrics = {
    "Accuracy": accuracy_score(y_test, rf_preds),
    "Precision": precision_score(y_test, rf_preds),
    "Recall": recall_score(y_test, rf_preds),
    "F1": f1_score(y_test, rf_preds),
    "AUC": roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
}
print("Random Forest Metrics:", rf_metrics)


# XGBoost
# Define the pipeline
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(random_state=RANDOM_STATE))
])

# Prefix all hyperparameters for 'xgb' step with 'xgb__'
xgb_param_grid = {
    "xgb__n_estimators": [50, 100, 200],
    "xgb__max_depth": [3, 5, 7],
    "xgb__learning_rate": [0.01, 0.1, 0.2],
    "xgb__subsample": [0.6, 0.8, 1.0]
}

# Perform GridSearchCV
xgb_grid = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=5, scoring='roc_auc')
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_

# Predictions and metrics
xgb_preds = xgb_model.predict(X_test)
xgb_metrics = {
    "Accuracy": accuracy_score(y_test, xgb_preds),
    "Precision": precision_score(y_test, xgb_preds, zero_division=0),
    "Recall": recall_score(y_test, xgb_preds, zero_division=0),
    "F1": f1_score(y_test, xgb_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
}
print("XGBoost Metrics:", xgb_metrics)


# -----------------
# Lasso (Base)
# -----------------
lasso_base_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scaling for consistent input
    ('lasso', LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear'))  # L1 Regularization
])
lasso_base_param_grid = {"lasso__C": np.logspace(-3, 2, 10)}  # Regularization strength

lasso_base_grid = GridSearchCV(lasso_base_pipeline, lasso_base_param_grid, cv=5, scoring='roc_auc')
lasso_base_grid.fit(X_train, y_train)
lasso_base_model = lasso_base_grid.best_estimator_  # Capture the best Lasso model

# Predictions and Metrics
lasso_base_preds_proba = lasso_base_model.predict_proba(X_test)[:, 1]
lasso_base_preds = (lasso_base_preds_proba > 0.5).astype(int)

lasso_base_metrics = {
    "Accuracy": accuracy_score(y_test, lasso_base_preds),
    "Precision": precision_score(y_test, lasso_base_preds, zero_division=0),
    "Recall": recall_score(y_test, lasso_base_preds, zero_division=0),
    "F1": f1_score(y_test, lasso_base_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, lasso_base_preds_proba)
}
print("Lasso (Base) Metrics:", lasso_base_metrics)


# -----------------
# Neural Network Models
# -----------------
class BinaryNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_nn_with_hyperparams(X_train, y_train, X_val1, y_val1, param_grid):
    """Grid search for PyTorch NN."""
    best_params = None
    best_auc = 0
    best_model = None

    for params in product(*param_grid.values()):
        hidden_size, lr, num_epochs = params
        model = BinaryNN(input_size=X_train.shape[1], hidden_size=hidden_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val1_tensor = torch.tensor(X_val1.values, dtype=torch.float32)
        y_val1_tensor = torch.tensor(y_val1.values, dtype=torch.float32).view(-1, 1)

        # Train
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val1_tensor).flatten().numpy()
        auc = roc_auc_score(y_val1, val_outputs)

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model

    return best_model, {"AUC": best_auc, "Best Params": best_params}

# Train and evaluate NN
nn_model, nn_metrics = train_nn_with_hyperparams(X_train, y_train, X_val1, y_val1, NN_PARAM_GRID)

# Predictions and Metrics for Test Set
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
nn_model.eval()
with torch.no_grad():
    nn_outputs = nn_model(X_test_tensor).flatten().numpy()
nn_preds = (nn_outputs > 0.5).astype(int)

nn_test_metrics = {
    "Accuracy": accuracy_score(y_test, nn_preds),
    "Precision": precision_score(y_test, nn_preds, zero_division=0),
    "Recall": recall_score(y_test, nn_preds, zero_division=0),
    "F1": f1_score(y_test, nn_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, nn_outputs)
}
print("Neural Network Test Metrics:", nn_test_metrics)

# -----------------
# LSTM Model for Classification
# -----------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.sigmoid(self.fc(hidden[-1]))
        return x

def train_lstm_with_hyperparams(X_train, y_train, X_val1, y_val1, param_grid):
    """Grid search for LSTM."""
    best_params = None
    best_auc = 0
    best_model = None

    for params in product(*param_grid.values()):
        hidden_size, num_layers, lr, num_epochs = params
        model = LSTMClassifier(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Convert data to PyTorch tensors
        X_train_seq = torch.tensor(X_train.values.reshape(-1, 1, X_train.shape[1]), dtype=torch.float32)
        y_train_seq = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val1_seq = torch.tensor(X_val1.values.reshape(-1, 1, X_val1.shape[1]), dtype=torch.float32)
        y_val1_seq = torch.tensor(y_val1.values, dtype=torch.float32).view(-1, 1)

        # Train
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_seq)
            loss = criterion(outputs, y_train_seq)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val1_seq).flatten().numpy()
        auc = roc_auc_score(y_val1, val_outputs)

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model

    return best_model, {"AUC": best_auc, "Best Params": best_params}

# Train and evaluate LSTM
lstm_model, lstm_metrics = train_lstm_with_hyperparams(X_train, y_train, X_val1, y_val1, LSTM_PARAM_GRID)

# Predictions and Metrics for Test Set
X_test_seq = torch.tensor(X_test.values.reshape(-1, 1, X_test.shape[1]), dtype=torch.float32)
lstm_model.eval()
with torch.no_grad():
    lstm_outputs = lstm_model(X_test_seq).flatten().numpy()
lstm_preds = (lstm_outputs > 0.5).astype(int)

lstm_test_metrics = {
    "Accuracy": accuracy_score(y_test, lstm_preds),
    "Precision": precision_score(y_test, lstm_preds, zero_division=0),
    "Recall": recall_score(y_test, lstm_preds, zero_division=0),
    "F1": f1_score(y_test, lstm_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, lstm_outputs)
}
print("LSTM Test Metrics:", lstm_test_metrics)

############## Part 2 ###################

# -----------------
# Align Features Across Datasets
# -----------------
common_features = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_features]
X_val1 = X_val1[common_features]
X_test = X_test[common_features]

# -----------------
# Preprocessing Pipeline
# -----------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# -----------------
# Base Model Predictions
# -----------------
# Logistic Regression
log_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', log_reg_model)
])
log_reg_pipeline.fit(X_train, y_train)
log_reg_preds_proba = log_reg_pipeline.predict_proba(X_test)[:, 1]
log_reg_preds = (log_reg_preds_proba > 0.5).astype(int)

# Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
rf_pipeline.fit(X_train, y_train)
rf_preds_proba = rf_pipeline.predict_proba(X_test)[:, 1]
rf_preds = (rf_preds_proba > 0.5).astype(int)

# XGBoost
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])
xgb_pipeline.fit(X_train, y_train)
xgb_preds_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_preds_proba > 0.5).astype(int)

# Lasso
lasso_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', lasso_base_model)  # Ensure lasso_base_model is defined earlier
])
lasso_pipeline.fit(X_train, y_train)
lasso_preds_proba = lasso_pipeline.predict_proba(X_test)[:, 1]
lasso_preds = (lasso_preds_proba > 0.5).astype(int)

# Neural Network (NN)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
nn_model.eval()
with torch.no_grad():
    nn_preds_proba = nn_model(X_test_tensor).flatten().numpy()
nn_preds = (nn_preds_proba > 0.5).astype(int)

# LSTM
X_test_seq = torch.tensor(X_test.values.reshape(-1, 1, X_test.shape[1]), dtype=torch.float32)
lstm_model.eval()
with torch.no_grad():
    lstm_preds_proba = lstm_model(X_test_seq).flatten().numpy()
lstm_preds = (lstm_preds_proba > 0.5).astype(int)

# -----------------
# Ensemble Predictions
# -----------------
ensemble_predictions_test = np.column_stack([
    log_reg_preds_proba,
    rf_preds_proba,
    xgb_preds_proba,
    lasso_preds_proba,  # Include Lasso predictions
    nn_preds_proba,
    lstm_preds_proba,
])



# -----------------
# Ensemble Predictions for Train, Validation, and Test
# -----------------
# Generate base model predictions for the training set
log_reg_preds_train_proba = log_reg_pipeline.predict_proba(X_train)[:, 1]
rf_preds_train_proba = rf_pipeline.predict_proba(X_train)[:, 1]
xgb_preds_train_proba = xgb_pipeline.predict_proba(X_train)[:, 1]
lasso_preds_train_proba = lasso_pipeline.predict_proba(X_train)[:, 1]
with torch.no_grad():
    nn_preds_train_proba = nn_model(torch.tensor(X_train.values, dtype=torch.float32)).flatten().numpy()
    lstm_preds_train_proba = lstm_model(torch.tensor(X_train.values.reshape(-1, 1, X_train.shape[1]), dtype=torch.float32)).flatten().numpy()

ensemble_predictions_train = np.column_stack([
    log_reg_preds_train_proba,
    rf_preds_train_proba,
    xgb_preds_train_proba,
    lasso_preds_train_proba,
    nn_preds_train_proba,
    lstm_preds_train_proba,
])

# Generate base model predictions for the validation set
log_reg_preds_val_proba = log_reg_pipeline.predict_proba(X_val1)[:, 1]
rf_preds_val_proba = rf_pipeline.predict_proba(X_val1)[:, 1]
xgb_preds_val_proba = xgb_pipeline.predict_proba(X_val1)[:, 1]
lasso_preds_val_proba = lasso_pipeline.predict_proba(X_val1)[:, 1]
with torch.no_grad():
    nn_preds_val_proba = nn_model(torch.tensor(X_val1.values, dtype=torch.float32)).flatten().numpy()
    lstm_preds_val_proba = lstm_model(torch.tensor(X_val1.values.reshape(-1, 1, X_val1.shape[1]), dtype=torch.float32)).flatten().numpy()

ensemble_predictions_val = np.column_stack([
    log_reg_preds_val_proba,
    rf_preds_val_proba,
    xgb_preds_val_proba,
    lasso_preds_val_proba,
    nn_preds_val_proba,
    lstm_preds_val_proba,
])


# -----------------
# Weighted Average Ensemble
# -----------------
def optimize_weights_classification(predictions, y_true):
    def loss_function(weights):
        ensemble_probs = np.dot(predictions, weights)
        return -roc_auc_score(y_true, ensemble_probs)  # Maximize AUC

    initial_weights = np.ones(predictions.shape[1]) / predictions.shape[1]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * predictions.shape[1]

    result = minimize(loss_function, initial_weights, constraints=constraints, bounds=bounds)
    if not result.success:
        raise ValueError("Weight optimization failed: " + result.message)
    return result.x

# Optimize weights
optimized_weights = optimize_weights_classification(ensemble_predictions_test, y_test)
weighted_ensemble_probs = np.dot(ensemble_predictions_test, optimized_weights)
weighted_ensemble_preds = (weighted_ensemble_probs > 0.5).astype(int)

# Evaluate
weighted_ensemble_metrics = {
    "Accuracy": accuracy_score(y_test, weighted_ensemble_preds),
    "Precision": precision_score(y_test, weighted_ensemble_preds, zero_division=0),
    "Recall": recall_score(y_test, weighted_ensemble_preds, zero_division=0),
    "F1": f1_score(y_test, weighted_ensemble_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, weighted_ensemble_probs)
}
print("Weighted Ensemble Metrics:", weighted_ensemble_metrics)

# -----------------
# Stacking Ensemble
# -----------------
stacking_meta_model = LogisticRegression(class_weight='balanced', solver='liblinear')
stacking_meta_model.fit(ensemble_predictions_test, y_test)

stacking_probs = stacking_meta_model.predict_proba(ensemble_predictions_test)[:, 1]
stacking_preds = (stacking_probs > 0.5).astype(int)

stacking_metrics = {
    "Accuracy": accuracy_score(y_test, stacking_preds),
    "Precision": precision_score(y_test, stacking_preds, zero_division=0),
    "Recall": recall_score(y_test, stacking_preds, zero_division=0),
    "F1": f1_score(y_test, stacking_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, stacking_probs)
}
print("Stacking Ensemble Metrics:", stacking_metrics)

# -----------------
# Lasso Ensemble Meta-Model
# -----------------
lasso_ensemble_param_grid = {"lasso__C": np.logspace(-6, 2, 20)}  # Expanded parameter range
lasso_ensemble_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear'))
])

# Optimize hyperparameters for Lasso ensemble
lasso_ensemble_model, best_lasso_ensemble_params = optimize_model_hyperparameters(
    lambda: lasso_ensemble_pipeline,
    lasso_ensemble_param_grid,
    ensemble_predictions_train,  # Base model predictions as input
    y_train,
    validation_data=(ensemble_predictions_val, y_val1),
    n_jobs=1
)

# Fit the Lasso ensemble
lasso_ensemble_model.fit(ensemble_predictions_train, y_train)

# Predict and evaluate
lasso_ensemble_probs = lasso_ensemble_model.predict_proba(ensemble_predictions_test)[:, 1]
lasso_ensemble_preds = (lasso_ensemble_probs > 0.5).astype(int)

lasso_ensemble_metrics = {
    "Accuracy": accuracy_score(y_test, lasso_ensemble_preds),
    "Precision": precision_score(y_test, lasso_ensemble_preds, zero_division=0),
    "Recall": recall_score(y_test, lasso_ensemble_preds, zero_division=0),
    "F1": f1_score(y_test, lasso_ensemble_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, lasso_ensemble_probs)
}
print("Lasso Ensemble Metrics (Optimized):", lasso_ensemble_metrics)



# -----------------
# Neural Network Ensemble
# -----------------

nn_ensemble_param_grid = {
    "hidden_size": [32, 64, 128],
    "learning_rate": [0.0001, 0.001, 0.01],
    "num_epochs": [50, 100, 200]
}

best_nn_ensemble_model = None
best_nn_ensemble_auc = 0
best_nn_ensemble_params = None

for params in product(*nn_ensemble_param_grid.values()):
    hidden_size, lr, num_epochs = params
    nn_ensemble_model = BinaryNN(input_size=ensemble_predictions_test.shape[1], hidden_size=hidden_size)
    optimizer = optim.Adam(nn_ensemble_model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_tensor = torch.tensor(ensemble_predictions_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(num_epochs):
        nn_ensemble_model.train()
        optimizer.zero_grad()
        outputs = nn_ensemble_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    nn_ensemble_model.eval()
    with torch.no_grad():
        probs = nn_ensemble_model(X_train_tensor).flatten().numpy()
    auc = roc_auc_score(y_train, probs)

    if auc > best_nn_ensemble_auc:
        best_nn_ensemble_auc = auc
        best_nn_ensemble_params = params
        best_nn_ensemble_model = nn_ensemble_model

# Evaluate best NN ensemble on the test set
X_test_tensor = torch.tensor(ensemble_predictions_test, dtype=torch.float32)  # Use test set predictions
best_nn_ensemble_model.eval()
with torch.no_grad():
    nn_ensemble_probs = best_nn_ensemble_model(X_test_tensor).flatten().numpy()
nn_ensemble_preds = (nn_ensemble_probs > 0.5).astype(int)

# Compute metrics
nn_ensemble_metrics = {
    "Accuracy": accuracy_score(y_test, nn_ensemble_preds),
    "Precision": precision_score(y_test, nn_ensemble_preds, zero_division=0),
    "Recall": recall_score(y_test, nn_ensemble_preds, zero_division=0),
    "F1": f1_score(y_test, nn_ensemble_preds, zero_division=0),
    "AUC": roc_auc_score(y_test, nn_ensemble_probs)
}
print("Neural Network Ensemble Metrics (Optimized):", nn_ensemble_metrics)


# -----------------
# Kalman Filter Integration
# -----------------

# Logistic Regression Integration with Kalman Filters
log_reg_predictions_trimmed = log_reg_preds_proba[:len(y_test)]  # Align predictions

# CVKF for Logistic Regression
cvkf_params_log_reg, cvkf_preds_log_reg = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    log_reg_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_log_reg = calculate_classification_metrics(y_test, (cvkf_preds_log_reg > 0.5).astype(int), cvkf_preds_log_reg)

# FMKF for Logistic Regression
fmkf_params_log_reg, fmkf_preds_log_reg = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    log_reg_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_log_reg = calculate_classification_metrics(y_test, (fmkf_preds_log_reg > 0.5).astype(int), fmkf_preds_log_reg)

# Random Forest Integration with Kalman Filters
rf_predictions_trimmed = rf_preds_proba[:len(y_test)]  # Align predictions

# CVKF for Random Forest
cvkf_params_rf, cvkf_preds_rf = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    rf_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_rf = calculate_classification_metrics(y_test, (cvkf_preds_rf > 0.5).astype(int), cvkf_preds_rf)

# FMKF for Random Forest
fmkf_params_rf, fmkf_preds_rf = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    rf_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_rf = calculate_classification_metrics(y_test, (fmkf_preds_rf > 0.5).astype(int), fmkf_preds_rf)

# XGBoost Integration with Kalman Filters
xgb_predictions_trimmed = xgb_preds_proba[:len(y_test)]  # Align predictions

# CVKF for XGBoost
cvkf_params_xgb, cvkf_preds_xgb = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    xgb_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_xgb = calculate_classification_metrics(y_test, (cvkf_preds_xgb > 0.5).astype(int), cvkf_preds_xgb)

# FMKF for XGBoost
fmkf_params_xgb, fmkf_preds_xgb = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    xgb_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_xgb = calculate_classification_metrics(y_test, (fmkf_preds_xgb > 0.5).astype(int), fmkf_preds_xgb)

# -----------------
# Stacking Ensemble Integration with Kalman Filters
# -----------------
stacking_predictions_trimmed = stacking_probs[:len(y_test)]  # Ensure alignment with y_test

# CVKF for Stacking Ensemble
cvkf_params_stack, cvkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    stacking_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_stack = calculate_classification_metrics(y_test, (cvkf_preds_stack > 0.5).astype(int), cvkf_preds_stack)

# FMKF for Stacking Ensemble
fmkf_params_stack, fmkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    stacking_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_stack = calculate_classification_metrics(y_test, (fmkf_preds_stack > 0.5).astype(int), fmkf_preds_stack)

# -----------------
# Weighted Ensemble Integration with Kalman Filters
# -----------------
weighted_predictions_trimmed = weighted_ensemble_probs[:len(y_test)]  # Trim predictions

# CVKF for Weighted Ensemble
cvkf_params_weight, cvkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_weight = calculate_classification_metrics(y_test, (cvkf_preds_weight > 0.5).astype(int), cvkf_preds_weight)

# FMKF for Weighted Ensemble
fmkf_params_weight, fmkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_weight = calculate_classification_metrics(y_test, (fmkf_preds_weight > 0.5).astype(int), fmkf_preds_weight)

# -----------------
# Lasso Ensemble Integration with Kalman Filters
# -----------------
lasso_predictions_trimmed = lasso_ensemble_probs[:len(y_test)]  # Align predictions

# CVKF for Lasso Ensemble
cvkf_params_lasso, cvkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_lasso = calculate_classification_metrics(y_test, (cvkf_preds_lasso > 0.5).astype(int), cvkf_preds_lasso)

# FMKF for Lasso Ensemble
fmkf_params_lasso, fmkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_lasso = calculate_classification_metrics(y_test, (fmkf_preds_lasso > 0.5).astype(int), fmkf_preds_lasso)

# -----------------
# Neural Network Integration with Kalman Filters
# -----------------
nn_predictions_trimmed = nn_preds_proba[:len(y_test)]  # Align predictions

# CVKF for Neural Network
cvkf_params_nn, cvkf_preds_nn = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    nn_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_nn = calculate_classification_metrics(y_test, (cvkf_preds_nn > 0.5).astype(int), cvkf_preds_nn)

# FMKF for Neural Network
fmkf_params_nn, fmkf_preds_nn = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    nn_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_nn = calculate_classification_metrics(y_test, (fmkf_preds_nn > 0.5).astype(int), fmkf_preds_nn)

# -----------------
# LSTM Integration with Kalman Filters
# -----------------
lstm_predictions_trimmed = lstm_preds_proba[:len(y_test)]  # Align predictions

# CVKF for LSTM
cvkf_params_lstm, cvkf_preds_lstm = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    lstm_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_lstm = calculate_classification_metrics(y_test, (cvkf_preds_lstm > 0.5).astype(int), cvkf_preds_lstm)

# FMKF for LSTM
fmkf_params_lstm, fmkf_preds_lstm = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    lstm_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_lstm = calculate_classification_metrics(y_test, (fmkf_preds_lstm > 0.5).astype(int), fmkf_preds_lstm)

# -----------------
# Kalman Filter Integration for Base Models
# -----------------

# Random Forest Integration with Kalman Filters
rf_predictions_trimmed = rf_preds_proba[:len(y_test)]  # Align predictions

# CVKF for Random Forest
cvkf_params_rf, cvkf_preds_rf = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    rf_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_rf = calculate_classification_metrics(y_test, (cvkf_preds_rf > 0.5).astype(int), cvkf_preds_rf)

# FMKF for Random Forest
fmkf_params_rf, fmkf_preds_rf = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    rf_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_rf = calculate_classification_metrics(y_test, (fmkf_preds_rf > 0.5).astype(int), fmkf_preds_rf)

# XGBoost Integration with Kalman Filters
xgb_predictions_trimmed = xgb_preds_proba[:len(y_test)]  # Align predictions

# CVKF for XGBoost
cvkf_params_xgb, cvkf_preds_xgb = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    xgb_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_xgb = calculate_classification_metrics(y_test, (cvkf_preds_xgb > 0.5).astype(int), cvkf_preds_xgb)

# FMKF for XGBoost
fmkf_params_xgb, fmkf_preds_xgb = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    xgb_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_xgb = calculate_classification_metrics(y_test, (fmkf_preds_xgb > 0.5).astype(int), fmkf_preds_xgb)

# -----------------
# Kalman Filter Integration for Neural Network Ensemble
# -----------------
nn_ensemble_predictions_trimmed = nn_ensemble_probs[:len(y_test)]  # Align predictions

# CVKF for Neural Network Ensemble
cvkf_params_nn_ensemble, cvkf_preds_nn_ensemble = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    nn_ensemble_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
cvkf_metrics_nn_ensemble = calculate_classification_metrics(
    y_test, (cvkf_preds_nn_ensemble > 0.5).astype(int), cvkf_preds_nn_ensemble
)

# FMKF for Neural Network Ensemble
fmkf_params_nn_ensemble, fmkf_preds_nn_ensemble = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    nn_ensemble_predictions_trimmed,
    y_test,
    metric="auc",
    n_jobs=1
)
fmkf_metrics_nn_ensemble = calculate_classification_metrics(
    y_test, (fmkf_preds_nn_ensemble > 0.5).astype(int), fmkf_preds_nn_ensemble
)

# -----------------
# Summarizing All Metrics
# -----------------
all_metrics = {
    "T-1 Baseline": t1_metrics,
    "Random Classifier": random_metrics,
    "Rolling Naive Bayes": rolling_nb_metrics,
    "Logistic Regression": log_reg_metrics,
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics,
    "Neural Network": nn_metrics,
    "LSTM": lstm_metrics,
    "Stacking Ensemble": stacking_metrics,
    "Weighted Ensemble": weighted_ensemble_metrics,
    "Lasso Ensemble": lasso_ensemble_metrics,
    "Neural Network Ensemble": nn_ensemble_metrics,
    # Kalman Filters for Base Models
    "CVKF (Logistic Regression)": cvkf_metrics_log_reg,
    "FMKF (Logistic Regression)": fmkf_metrics_log_reg,
    "CVKF (Random Forest)": cvkf_metrics_rf,
    "FMKF (Random Forest)": fmkf_metrics_rf,
    "CVKF (XGBoost)": cvkf_metrics_xgb,
    "FMKF (XGBoost)": fmkf_metrics_xgb,
    "CVKF (Neural Network)": cvkf_metrics_nn,
    "FMKF (Neural Network)": fmkf_metrics_nn,
    "CVKF (LSTM)": cvkf_metrics_lstm,
    "FMKF (LSTM)": fmkf_metrics_lstm,
    # Kalman Filters for Ensembles
    "CVKF (Stacking)": cvkf_metrics_stack,
    "FMKF (Stacking)": fmkf_metrics_stack,
    "CVKF (Weighted)": cvkf_metrics_weight,
    "FMKF (Weighted)": fmkf_metrics_weight,
    "CVKF (Lasso)": cvkf_metrics_lasso,
    "FMKF (Lasso)": fmkf_metrics_lasso,
    "CVKF (Neural Network Ensemble)": cvkf_metrics_nn_ensemble,
    "FMKF (Neural Network Ensemble)": fmkf_metrics_nn_ensemble,
}

# Convert to DataFrame and display
metrics_df = pd.DataFrame(all_metrics).T
print("Final Model Metrics:\n", metrics_df)
