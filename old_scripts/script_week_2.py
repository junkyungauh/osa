from preprocessing.preprocessing import normalize_data, create_lagged_features
from models.models import linear_regression_model, logistic_regression_model, train_lstm_model
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics
from visualizations.visualizations import plot_regression_performance, plot_classification_performance
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import torch

# File paths
cont_data_path = 'simulated_series_cont.csv'
bin_data_path = 'simulated_series_bin.csv'

# Load Continuous Data
cont_data = pd.read_csv(cont_data_path)
cont_data.set_index('time', inplace=True)

# Preprocessing Continuous Data
cont_data = normalize_data(cont_data + 1, method="logistic")
cont_data = create_lagged_features(cont_data, column="return", lag=1)
X_cont = cont_data.filter(like='sig_')
y_cont = cont_data['return']

# Split Continuous Data
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(X_cont, y_cont, test_size=0.2, random_state=42)

# Align indices for ARIMA
y_train_cont.index = pd.RangeIndex(start=0, stop=len(y_train_cont))
y_test_cont.index = pd.RangeIndex(start=len(y_train_cont), stop=len(y_train_cont) + len(y_test_cont))

# Train Linear Regression Model
linear_model = linear_regression_model(X_train_cont, y_train_cont)
y_pred_lr = linear_model.predict(X_test_cont)

# Train ARIMA Model
try:
    arima_model = ARIMA(y_train_cont.values, order=(1, 1, 1))
    arima_fitted = arima_model.fit()
    y_pred_arima = arima_fitted.predict(
        start=len(y_train_cont), 
        end=len(y_train_cont) + len(y_test_cont) - 1
    )
    y_pred_arima = pd.Series(y_pred_arima, index=y_test_cont.index)
except Exception as e:
    print("ARIMA Model encountered an issue:", e)
    y_pred_arima = None

# Evaluate Models
lr_metrics = calculate_regression_metrics(y_test_cont, y_pred_lr)
arima_metrics = calculate_regression_metrics(y_test_cont, y_pred_arima) if y_pred_arima is not None else None

print("Linear Regression Metrics:", lr_metrics)
if arima_metrics:
    print("ARIMA Metrics:", arima_metrics)

# Train LSTM Model
X_train_lstm = X_train_cont.values.reshape(-1, 1, X_train_cont.shape[1])
y_train_cont_tensor = torch.tensor(y_train_cont.values, dtype=torch.float32).view(-1, 1)
lstm_model = train_lstm_model(X_train_lstm, y_train_cont_tensor, input_size=X_train_cont.shape[1])

# Ensure proper tensor conversion for testing
X_test_lstm = torch.tensor(X_test_cont.values.reshape(-1, 1, X_test_cont.shape[1]), dtype=torch.float32)
y_pred_lstm = lstm_model(X_test_lstm).detach().cpu().numpy().flatten()

# Evaluate LSTM
lstm_metrics = calculate_regression_metrics(y_test_cont, y_pred_lstm)
print("LSTM Metrics:", lstm_metrics)

# Regression Ensemble
def weighted_average_ensemble(y_true, preds_dict):
    weights = {model: 1 / calculate_regression_metrics(y_true, preds)["MSE"] for model, preds in preds_dict.items()}
    weights_sum = sum(weights.values())
    weighted_preds = sum(preds * (weight / weights_sum) for preds, weight in zip(preds_dict.values(), weights.values()))
    return weighted_preds

regression_preds = {"Linear Regression": y_pred_lr, "LSTM": y_pred_lstm}
if y_pred_arima is not None:
    regression_preds["ARIMA"] = y_pred_arima
ensemble_pred = weighted_average_ensemble(y_test_cont, regression_preds)
ensemble_metrics = calculate_regression_metrics(y_test_cont, ensemble_pred)
print("Regression Ensemble Metrics:", ensemble_metrics)

# Visualizations for Regression
regression_metrics = {
    "Linear Regression": lr_metrics,
    "LSTM": lstm_metrics,
    "Ensemble": ensemble_metrics,
}
if arima_metrics:
    regression_metrics["ARIMA"] = arima_metrics
plot_regression_performance(regression_metrics)

# Load Binary Data
bin_data = pd.read_csv(bin_data_path)
bin_data.set_index('time', inplace=True)

# Preprocessing Binary Data
bin_data = normalize_data(bin_data + 1, method="logistic")
bin_data['binary_category'] = (bin_data['binary'] > bin_data['binary'].median()).astype(int)

X_bin = bin_data.filter(like='sig_')
y_bin = bin_data['binary_category']

# Split Binary Data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# Train Logistic Regression
logistic_model = logistic_regression_model(X_train_bin, y_train_bin)
y_pred_log = logistic_model.predict(X_test_bin)

# Evaluate Logistic Regression
log_metrics = calculate_classification_metrics(y_test_bin, y_pred_log)
print("Logistic Regression Metrics:", log_metrics)

# Visualizations for Classification
plot_classification_performance({"Logistic Regression": log_metrics})
