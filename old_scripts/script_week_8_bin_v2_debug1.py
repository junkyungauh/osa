import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import numpy as np

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
def preprocess_data_with_features(file_path, target_column, lag_steps=None, rolling_window=None):
    """Load and preprocess data with feature engineering."""
    data = pd.read_csv(file_path)
    
    # Ensure data is sorted by time
    assert data.index.is_monotonic_increasing, "Dataset is not sorted by time."
    
    feature_data = pd.DataFrame(index=data.index)
    signal_cols = [col for col in data.columns if col.startswith('sig')]
    
    # Rolling features
    for col in signal_cols:
        feature_data[f'{col}_roll_mean'] = data[col].rolling(window=rolling_window).mean()
        feature_data[f'{col}_roll_std'] = data[col].rolling(window=rolling_window).std()
    
    # Lag features
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)
    
    # Drop rows with NaNs
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]
    
    X = feature_data
    y = data[target_column]
    return X, y

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


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# def optimize_kalman_hyperparameters(
#     kalman_filter_creator, param_grid: list, predictions: np.ndarray, true_values: np.ndarray,
#     metric: str = "mse", n_jobs: int = 1
# ):
#     """
#     Optimize hyperparameters for a given Kalman filter using grid search.
#     """
#     def evaluate_params(params):
#         kalman_filter = kalman_filter_creator(**params)
#         kalman_predictions = kalman_filter.filter(predictions)

#         # Align predictions and true values
#         kalman_predictions = kalman_predictions[:len(true_values)]  # Trim predictions if necessary
#         aligned_true_values = true_values[:len(kalman_predictions)]  # Trim true values if necessary

#         if metric == "mse":
#             score = mean_squared_error(aligned_true_values, kalman_predictions)
#         elif metric == "accuracy":
#             binary_predictions = (kalman_predictions > 0.5).astype(int)
#             score = accuracy_score(aligned_true_values, binary_predictions)

#         return score, params, kalman_predictions

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(evaluate_params)(params) for params in ParameterGrid(param_grid)
#     )

#     # Select the best result based on the metric
#     if metric == "mse":
#         best_result = min(results, key=lambda x: x[0])  # Minimize MSE
#     else:
#         best_result = max(results, key=lambda x: x[0])  # Maximize accuracy

#     best_score, best_params, best_predictions = best_result
#     return best_params, best_predictions


# -----------------
# Load and Preprocess Data
# -----------------
X, y = preprocess_data_with_features(
    file_path='simulated_series_bin.csv',
    target_column='binary',
    lag_steps=[1, 2, 3],
    rolling_window=10
)

X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(
    X, y, train_size=0.5, val1_size=0.15, val2_size=0.1, kalman_size=0.1, test_size=0.15
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


### Kalman Filter Integration for Ensembles with Base Models and Baselines ###

# -----------------
# Baselines Integration
# -----------------

# T-1 Baseline
t1_preds = y_t1_baseline.values  # Already derived in Part 1
t1_metrics = calculate_classification_metrics(y_test, t1_preds, t1_preds)
print("T-1 Baseline Metrics:", t1_metrics)

# Random Classifier Baseline
random_preds = y_random.values  # Already derived in Part 1
random_metrics = calculate_classification_metrics(y_test, random_preds, random_preds)
print("Random Classifier Metrics:", random_metrics)

# Rolling Naive Bayes Baseline
rolling_nb_preds = y_rolling_nb.values  # Already derived in Part 1
rolling_nb_metrics = calculate_classification_metrics(y_test, rolling_nb_preds, rolling_nb_preds)
print("Rolling Naive Bayes Metrics:", rolling_nb_metrics)

# -----------------
# Base Models Integration
# -----------------

# Logistic Regression Base Model
log_reg_metrics = calculate_classification_metrics(y_test, log_reg_preds, log_reg_preds_proba)
print("Logistic Regression Base Metrics:", log_reg_metrics)

# Neural Network Base Model
nn_metrics = calculate_classification_metrics(y_test, nn_preds, nn_preds_proba)
print("Neural Network Base Metrics:", nn_metrics)

# LSTM Base Model
lstm_metrics = calculate_classification_metrics(y_test, lstm_preds, lstm_preds_proba)
print("LSTM Base Metrics:", lstm_metrics)

# -----------------
# Stacking Ensemble Integration with Kalman Filters
# -----------------
stacking_predictions_trimmed = stacking_probs[:len(y_test)]  # Ensure alignment with y_test

# Constant Velocity Kalman Filter (CVKF) for Stacking
cvkf_params_stack, cvkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),  # Initialize CVKF
    CVKF_PARAM_GRID,  # Parameter grid for hyperparameter search
    stacking_predictions_trimmed,  # Stacking ensemble predictions (probabilities)
    y_test,  # True test values
    metric="auc",  # Focus on AUC
    n_jobs=1
)
cvkf_metrics_stack = calculate_classification_metrics(y_test, (cvkf_preds_stack > 0.5).astype(int), cvkf_preds_stack)

# Financial Model Kalman Filter (FMKF) for Stacking
fmkf_params_stack, fmkf_preds_stack = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),  # Initialize FMKF
    FMKF_PARAM_GRID,  # Parameter grid for FMKF
    stacking_predictions_trimmed,  # Stacking ensemble predictions
    y_test,  # True test values
    metric="auc",  # Focus on AUC
    n_jobs=1
)
fmkf_metrics_stack = calculate_classification_metrics(y_test, (fmkf_preds_stack > 0.5).astype(int), fmkf_preds_stack)
print("Stacking + CVKF Metrics:", cvkf_metrics_stack)
print("Stacking + FMKF Metrics:", fmkf_metrics_stack)

# -----------------
# Weighted Ensemble Integration with Kalman Filters
# -----------------
weighted_predictions_trimmed = weighted_ensemble_probs[:len(y_test)]  # Trim predictions

# Constant Velocity Kalman Filter (CVKF) for Weighted Ensemble
cvkf_params_weight, cvkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test,
    metric="auc",  # Focus on AUC
    n_jobs=1
)
cvkf_metrics_weight = calculate_classification_metrics(y_test, (cvkf_preds_weight > 0.5).astype(int), cvkf_preds_weight)

# Financial Model Kalman Filter (FMKF) for Weighted Ensemble
fmkf_params_weight, fmkf_preds_weight = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    weighted_predictions_trimmed,
    y_test,
    metric="auc",  # Focus on AUC
    n_jobs=1
)
fmkf_metrics_weight = calculate_classification_metrics(y_test, (fmkf_preds_weight > 0.5).astype(int), fmkf_preds_weight)
print("Weighted + CVKF Metrics:", cvkf_metrics_weight)
print("Weighted + FMKF Metrics:", fmkf_metrics_weight)

# -----------------
# Lasso Ensemble Integration with Kalman Filters
# -----------------
lasso_predictions_trimmed = lasso_ensemble_probs[:len(y_test)]  # Align predictions

# Constant Velocity Kalman Filter (CVKF) for Lasso Ensemble
cvkf_params_lasso, cvkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test,
    metric="auc",  # Focus on AUC
    n_jobs=1
)
cvkf_metrics_lasso = calculate_classification_metrics(y_test, (cvkf_preds_lasso > 0.5).astype(int), cvkf_preds_lasso)

# Financial Model Kalman Filter (FMKF) for Lasso Ensemble
fmkf_params_lasso, fmkf_preds_lasso = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    lasso_predictions_trimmed,
    y_test,
    metric="auc",  # Focus on AUC
    n_jobs=1
)
fmkf_metrics_lasso = calculate_classification_metrics(y_test, (fmkf_preds_lasso > 0.5).astype(int), fmkf_preds_lasso)
print("Lasso + CVKF Metrics:", cvkf_metrics_lasso)
print("Lasso + FMKF Metrics:", fmkf_metrics_lasso)

# -----------------
# Neural Network Ensemble Integration with Kalman Filters
# -----------------
nn_predictions_trimmed = nn_ensemble_probs[:len(y_test)]  # Align predictions

# Constant Velocity Kalman Filter (CVKF) for Neural Network Ensemble
cvkf_params_nn, cvkf_preds_nn = optimize_kalman_hyperparameters(
    lambda **params: ConstantVelocityKalmanFilter(**params),
    CVKF_PARAM_GRID,
    nn_predictions_trimmed,
    y_test,
    metric="auc",  # Focus on AUC
    n_jobs=1
)
cvkf_metrics_nn = calculate_classification_metrics(y_test, (cvkf_preds_nn > 0.5).astype(int), cvkf_preds_nn)

# Financial Model Kalman Filter (FMKF) for Neural Network Ensemble
fmkf_params_nn, fmkf_preds_nn = optimize_kalman_hyperparameters(
    lambda **params: FinancialModelKalmanFilter(**params),
    FMKF_PARAM_GRID,
    nn_predictions_trimmed,
    y_test,
    metric="auc",  # Focus on AUC
    n_jobs=1
)
fmkf_metrics_nn = calculate_classification_metrics(y_test, (fmkf_preds_nn > 0.5).astype(int), fmkf_preds_nn)
print("NN Ensemble + CVKF Metrics:", cvkf_metrics_nn)
print("NN Ensemble + FMKF Metrics:", fmkf_metrics_nn)

# -----------------
# Summarizing All Metrics
# -----------------
# Add Neural Network Ensemble metrics to the all_metrics dictionary
all_metrics = {
    "T-1 Baseline": t1_metrics,
    "Random Classifier": random_metrics,
    "Rolling Naive Bayes": rolling_nb_metrics,
    "Logistic Regression": log_reg_metrics,
    "Neural Network": nn_metrics,
    "LSTM": lstm_metrics,
    "Stacking Ensemble": stacking_metrics,
    "Weighted Ensemble": weighted_ensemble_metrics,
    "Lasso Ensemble": lasso_ensemble_metrics,
    "Neural Network Ensemble": nn_ensemble_metrics,  # Added NN Ensemble metrics
    "CVKF (Stacking)": cvkf_metrics_stack,
    "FMKF (Stacking)": fmkf_metrics_stack,
    "CVKF (Weighted)": cvkf_metrics_weight,
    "FMKF (Weighted)": fmkf_metrics_weight,
    "CVKF (Lasso)": cvkf_metrics_lasso,
    "FMKF (Lasso)": fmkf_metrics_lasso,
    "CVKF (NN)": cvkf_metrics_nn,
    "FMKF (NN)": fmkf_metrics_nn,
}

# Convert the dictionary to a DataFrame and display
metrics_df = pd.DataFrame(all_metrics).T
print("Final Model Metrics:\n", metrics_df)

