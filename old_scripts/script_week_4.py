from preprocessing.preprocessing import normalize_data, create_lagged_features
from models.models import linear_regression_model, logistic_regression_model
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics, calculate_regression_information_gain, calculate_classification_information_gain
from visualizations.visualizations import plot_error_trend, plot_tsne_clusters, plot_elbow_method
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
cont_data_path = 'simulated_series_cont.csv'
bin_data_path = 'simulated_series_bin.csv'

# Load Datasets
cont_data = pd.read_csv(cont_data_path)
bin_data = pd.read_csv(bin_data_path)

# Preprocessing Continuous Data
cont_data = create_lagged_features(cont_data, column="return", lag=1)
cont_data = normalize_data(cont_data, method="z-score")
X_cont = cont_data.drop(columns=["return"])
y_cont = cont_data["return"]

# Train Linear Regression for SHAP Analysis
reg_model = linear_regression_model(X_cont, y_cont)

# SHAP Analysis
explainer = shap.LinearExplainer(reg_model, X_cont)
shap_values = explainer.shap_values(X_cont)
shap_df = pd.DataFrame(np.abs(shap_values), columns=X_cont.columns)

# Regression Metrics and Information Gain
reg_metrics = calculate_regression_metrics(y_cont, reg_model.predict(X_cont))
reg_info_gain = calculate_regression_information_gain(y_cont, reg_model.predict(X_cont))
print("Regression Metrics:", reg_metrics)
print("Regression Information Gain:", reg_info_gain)

# Classification Workflow
bin_data = create_lagged_features(bin_data, column="binary", lag=1)
bin_data = normalize_data(bin_data, method="z-score")
bin_data["binary"] = (bin_data["binary"] > bin_data["binary"].median()).astype(int)
X_bin = bin_data.drop(columns=["binary"])
y_bin = bin_data["binary"]

# Logistic Regression for Classification
logistic_model = logistic_regression_model(X_bin, y_bin)
logistic_preds = logistic_model.predict(X_bin)
logistic_pred_proba = logistic_model.predict_proba(X_bin)
classification_metrics = calculate_classification_metrics(y_bin, logistic_preds)
classification_info_gain = calculate_classification_information_gain(y_bin, logistic_pred_proba)
print("Classification Metrics:", classification_metrics)
print("Classification Information Gain:", classification_info_gain)
