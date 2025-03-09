from preprocessing.preprocessing import normalize_data, create_lagged_features
from models.models import linear_regression_model, logistic_regression_model
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics
from utils.utils import load_data, calculate_rolling_binary_metrics
from visualizations.visualizations import plot_regression_performance, plot_classification_performance
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data using the utility function
cont_data_path = 'simulated_series_cont.csv'
cont_data = load_data(cont_data_path, index_col='time')

# Display summary and handle missing values
print("Data Summary:")
print(cont_data.head())
print("Number of NaNs in each column:\n", cont_data.isnull().sum())

cont_data = cont_data.dropna(subset=['return'])  # Clean data by dropping rows with NaN in 'return'

# Normalize and preprocess continuous data
cont_data = normalize_data(cont_data + 1, method="logistic")
cont_data = create_lagged_features(cont_data, column="return", lag=1)

# Regression task setup
X_cont = cont_data.drop(columns=["return"])
y_cont = cont_data["return"]
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(
    X_cont, y_cont, test_size=0.2, random_state=42
)

# Baseline Model (T-1)
y_baseline = y_cont.shift(1).fillna(method="bfill")
baseline_metrics = calculate_regression_metrics(y_test_cont, y_baseline[-len(y_test_cont):])
print("Baseline Metrics:", baseline_metrics)

# Linear Regression (Single Feature)
linear_single_model = linear_regression_model(X_train_cont[["return_lag1"]], y_train_cont)
y_pred_single = linear_single_model.predict(X_test_cont[["return_lag1"]])
single_metrics = calculate_regression_metrics(y_test_cont, y_pred_single)
print("Single Feature Regression Metrics:", single_metrics)

# Linear Regression (Multi-Feature)
linear_multi_model = linear_regression_model(X_train_cont, y_train_cont)
y_pred_multi = linear_multi_model.predict(X_test_cont)
multi_metrics = calculate_regression_metrics(y_test_cont, y_pred_multi)
print("Multi-Feature Regression Metrics:", multi_metrics)

# Plot Regression Performance
plot_regression_performance({
    "Baseline": baseline_metrics,
    "Linear Regression (Single Feature)": single_metrics,
    "Linear Regression (Multi-Feature)": multi_metrics
})

# Binary Classification Task
cont_data["return_binary"] = (cont_data["return"] > cont_data["return"].median()).astype(int)
X_bin = cont_data.drop(columns=["return", "return_binary"])
y_bin = cont_data["return_binary"]
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42
)

# Logistic Regression Model
logistic_model = logistic_regression_model(X_train_bin, y_train_bin, max_iter=1000)
y_pred_log = logistic_model.predict(X_test_bin)
classification_metrics = calculate_classification_metrics(y_test_bin, y_pred_log)
print("Logistic Regression Metrics:", classification_metrics)

# Rolling Metrics for Binary Classification
rolling_accuracy, rolling_f1 = calculate_rolling_binary_metrics(y_test_bin, y_pred_log, window=10)

# Plot Rolling Metrics
plt.plot(rolling_accuracy, label="Rolling Accuracy")
plt.plot(rolling_f1, label="Rolling F1-Score", color="orange")
plt.title("Rolling Metrics (Window=10)")
plt.xlabel("Time")
plt.ylabel("Score")
plt.legend()
plt.show()

# Plot Classification Performance
plot_classification_performance({"Logistic Regression": classification_metrics})
