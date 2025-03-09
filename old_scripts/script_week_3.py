from preprocessing.preprocessing import normalize_data, create_lagged_features, split_temporal_data
from models.models import linear_regression_model, logistic_regression_model
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics
from visualizations.visualizations import plot_regression_performance, plot_classification_performance
from utils.utils import calculate_rolling_continuous_metrics, calculate_rolling_binary_metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
cont_data_path = 'simulated_series_cont.csv'
bin_data_path = 'simulated_series_bin.csv'

# Load Continuous and Binary Data
cont_data = pd.read_csv(cont_data_path)
bin_data = pd.read_csv(bin_data_path)

# Preprocessing Continuous Data
cont_data = create_lagged_features(cont_data, column="return", lag=1)
cont_data = normalize_data(cont_data, method="logistic")
X_cont = cont_data.drop(columns=["return"])
y_cont = cont_data["return"]

# Preprocessing Binary Data
bin_data = create_lagged_features(bin_data, column="binary", lag=1)

# Normalize feature columns only
X_bin = bin_data.drop(columns=["binary"])
X_bin = normalize_data(X_bin, method="logistic")

# Ensure binary column is truly binary (0/1)
y_bin = (bin_data["binary"] > bin_data["binary"].median()).astype(int)

# Confirm y_bin is binary
print("Unique values in y_bin:", y_bin.unique())

# TimeSeriesSplit for Regression
tscv = TimeSeriesSplit(n_splits=5)

# Baseline Model (T-1 for Regression)
y_cont_baseline = y_cont.shift(1).fillna(method="bfill")
baseline_metrics = calculate_regression_metrics(y_cont, y_cont_baseline)
print("Baseline (T-1) Metrics:", baseline_metrics)

# Linear Regression (Autoregressive Model)
auto_reg_model = linear_regression_model(X_cont[["return_lag1"]], y_cont)
auto_reg_metrics = calculate_regression_metrics(y_cont, auto_reg_model.predict(X_cont[["return_lag1"]]))
print("Autoregressive Model Metrics:", auto_reg_metrics)

# Multi-Feature Linear Regression
multi_model = linear_regression_model(X_cont, y_cont)
multi_model_metrics = calculate_regression_metrics(y_cont, multi_model.predict(X_cont))
print("Multi-Feature Regression Metrics:", multi_model_metrics)

# Classification (Logistic Regression with TimeSeriesSplit)
classification_model = logistic_regression_model(X_bin, y_bin)
classification_metrics = calculate_classification_metrics(y_bin, classification_model.predict(X_bin))
print("Logistic Regression Metrics:", classification_metrics)

# Visualizations
plot_regression_performance({
    "T-1 Baseline": baseline_metrics,
    "Autoregressive Model": auto_reg_metrics,
    "Multi-Feature Regression": multi_model_metrics
})
plot_classification_performance({"Logistic Regression": classification_metrics})

# SHAP Analysis for Regression
explainer_reg = shap.LinearExplainer(multi_model, X_cont)
shap_values_reg = explainer_reg.shap_values(X_cont)
shap.summary_plot(shap_values_reg, X_cont)

# Rolling SHAP Analysis
def calculate_rolling_shap(X, model, window_size=10):
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(np.abs(shap_values), columns=X.columns)
    return shap_df.rolling(window=window_size).mean().dropna()

rolling_shap = calculate_rolling_shap(X_cont, multi_model, window_size=10)
sns.heatmap(rolling_shap.T, cmap="viridis", cbar=True)
plt.title("Time-Series Rolling SHAP Values (Window=10)")
plt.show()

# Permutation Importance
perm_importance = permutation_importance(multi_model, X_cont, y_cont, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame(perm_importance.importances_mean, index=X_cont.columns, columns=["Importance"])
perm_importance_df.sort_values(by="Importance", ascending=False).plot(kind="barh", figsize=(10, 6))
plt.title("Permutation Feature Importance")
plt.xlabel("Importance (Mean Decrease in MAE)")
plt.show()

# Partial Dependence Plots for Regression
PartialDependenceDisplay.from_estimator(multi_model, X_cont, features=list(X_cont.columns)[:3], grid_resolution=50)
plt.suptitle("Partial Dependence Plots for Regression")
plt.show()

# Rolling Metrics for Classification
rolling_accuracy, rolling_f1 = calculate_rolling_binary_metrics(
    y_bin, classification_model.predict(X_bin), window=10
)
plt.plot(rolling_accuracy, label="Rolling Accuracy")
plt.plot(rolling_f1, label="Rolling F1-Score", color="orange")
plt.title("Rolling Metrics (Window=10)")
plt.xlabel("Time")
plt.ylabel("Score")
plt.legend()
plt.show()

# Rolling Metrics for Continuous Regression
rolling_mse, rolling_mae = calculate_rolling_continuous_metrics(
    y_cont, multi_model.predict(X_cont), window=10
)
plt.plot(rolling_mse, label="Rolling MSE")
plt.plot(rolling_mae, label="Rolling MAE", color="orange")
plt.title("Rolling Regression Metrics (Window=10)")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()
plt.show()
