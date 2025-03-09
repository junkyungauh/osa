from preprocessing.preprocessing import normalize_data, create_lagged_features, split_temporal_data
from models.models import linear_regression_model, logistic_regression_model
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics
from visualizations.visualizations import plot_regression_performance, plot_classification_performance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# File paths (update to actual paths)
cont_data_path = '/content/sample_data/01_aggregator/simulated_series_cont.csv'
bin_data_path = '/content/sample_data/01_aggregator/simulated_series_bin.csv'

# Load and preprocess data
data = pd.read_csv(cont_data_path)
data.set_index('time', inplace=True)
nan_counts = data.isnull().sum()
print("Number of NaNs in each column:\\n", nan_counts)

# Drop rows with NaN values in the 'return' column
data_clean = data.dropna(subset=['return'])

# Normalize signals using log transformation
data_clean = normalize_data(data_clean + 1, method="logistic")

# Generate lagged features
data_clean = create_lagged_features(data_clean, column="return", lag=1)

# Define features and target
signal_columns = [col for col in data_clean.columns if 'sig_' in col]
X = data_clean[signal_columns]
y = data_clean['return']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline 1: T-1 Model
data_clean['t-1'] = data_clean['return'].shift(1)
data_clean['t-1_predicted'] = data_clean['t-1']
y_pred_t1 = data_clean['t-1_predicted'].fillna(method='bfill')

# Evaluate T-1 Model
t1_metrics = calculate_regression_metrics(y_test, y_pred_t1[-len(y_test):])
print("T-1 Model Metrics:", t1_metrics)

# Baseline 2: Linear Regression Model
linear_model = linear_regression_model(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)

# Evaluate Linear Regression
lr_metrics = calculate_regression_metrics(y_test, y_pred_lr)
print("Linear Regression Metrics:", lr_metrics)

# Baseline 3: Logistic Regression Model
data_clean['return_binary'] = (data_clean['return'] > data_clean['t-1']).astype(int)
data_clean = data_clean.dropna(subset=['return_binary'])
X_log = data_clean[signal_columns]
y_log = data_clean['return_binary']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

logistic_model = logistic_regression_model(X_train_log, y_train_log)
y_pred_log = logistic_model.predict(X_test_log)

# Evaluate Logistic Regression
log_metrics = calculate_classification_metrics(y_test_log, y_pred_log)
print("Logistic Regression Metrics:", log_metrics)

# Visualization
plot_regression_performance({"T-1 Model": t1_metrics, "Linear Regression": lr_metrics})
plot_classification_performance({"Logistic Regression": log_metrics})


# Binary Data Processing and Analysis

# Load binary data
binary_data = pd.read_csv(bin_data_path)
binary_data.set_index('time', inplace=True)

# Drop rows with NaN values
binary_data_clean = binary_data.dropna()

# Normalize the binary data
binary_data_clean = normalize_data(binary_data_clean + 1, method="logistic")

# Create binary labels (e.g., classify based on a threshold of 'return')
binary_data_clean['return_binary'] = (binary_data_clean['return'] > binary_data_clean['return'].median()).astype(int)

# Define features and binary target
signal_columns_bin = [col for col in binary_data_clean.columns if 'sig_' in col]
X_bin = binary_data_clean[signal_columns_bin]
y_bin = binary_data_clean['return_binary']

# Split into train and test sets
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# Train a logistic regression model
logistic_model_bin = logistic_regression_model(X_train_bin, y_train_bin)
y_pred_bin = logistic_model_bin.predict(X_test_bin)

# Evaluate the logistic regression model
bin_metrics = calculate_classification_metrics(y_test_bin, y_pred_bin)
print("Binary Logistic Regression Metrics:", bin_metrics)

# Visualization for binary model performance
plot_classification_performance({"Binary Logistic Regression": bin_metrics})
