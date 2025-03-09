from preprocessing.preprocessing import normalize_data, create_lagged_features
from models.models import linear_regression_model, logistic_regression_model, kalman_filter_model
from metrics.metrics import calculate_regression_metrics, calculate_classification_metrics, calculate_information_gain
from visualizations.visualizations import plot_error_trend, plot_tsne_clusters, plot_elbow_method
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
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
cont_data = create_lagged_features(cont_data, target_column="return", lag=1)
cont_data = normalize_data(cont_data, method="z-score")
X_cont = cont_data.drop(columns=["return"])
y_cont = cont_data["return"]

# Train Linear Regression for SHAP Analysis
reg_model = linear_regression_model(X_cont, y_cont)

# SHAP Analysis
explainer = shap.LinearExplainer(reg_model, X_cont)
shap_values = explainer.shap_values(X_cont)
shap_df = pd.DataFrame(np.abs(shap_values), columns=X_cont.columns)

# Elbow Method for KMeans Clustering
plot_elbow_method(shap_df, max_clusters=10, title="Elbow Method for Optimal Clusters")

# Rolling SHAP Values and Clustering
rolling_shap = shap_df.rolling(window=10).mean().dropna()
kmeans = KMeans(n_clusters=4, random_state=42)
rolling_shap['Cluster'] = kmeans.fit_predict(rolling_shap)

# Heatmap for Rolling SHAP
sns.heatmap(rolling_shap.drop(columns="Cluster").T, cmap="coolwarm", cbar=True)
plt.title("Rolling SHAP Values with Clustering")
plt.show()

# t-SNE Visualization
plot_tsne_clusters(rolling_shap, cluster_labels=rolling_shap["Cluster"], title="t-SNE Clusters (Regression)")

# Static Kalman Filter for Regression
kalman_weights = kmeans.cluster_centers_[0]
kf_model = kalman_filter_model(initial_weights=kalman_weights)
kalman_predictions = []
for i in range(len(X_cont)):
    pred = np.dot(X_cont.iloc[i].values, kf_model.weights)
    kalman_predictions.append(pred)
    error = y_cont.iloc[i] - pred
    kf_model.update(measurement=error * X_cont.iloc[i].values)

# Error Statistics for Kalman Filter
kalman_metrics = calculate_regression_metrics(y_cont, kalman_predictions)
print("Kalman Filter Metrics:", kalman_metrics)

# Classification Workflow
bin_data = create_lagged_features(bin_data, target_column="binary", lag=1)
bin_data = normalize_data(bin_data, method="z-score")
X_bin = bin_data.drop(columns=["binary"])
y_bin = bin_data["binary"]

# Logistic Regression for Classification
logistic_model = logistic_regression_model(X_bin, y_bin)
logistic_preds = logistic_model.predict(X_bin)
classification_metrics = calculate_classification_metrics(y_bin, logistic_preds)
print("Logistic Regression Metrics:", classification_metrics)

# SHAP Analysis for Classification
explainer_cls = shap.LinearExplainer(logistic_model, X_bin)
shap_values_cls = explainer_cls.shap_values(X_bin)
shap_df_cls = pd.DataFrame(np.abs(shap_values_cls), columns=X_bin.columns)

# t-SNE for Classification Clusters
kmeans_cls = KMeans(n_clusters=4, random_state=42)
shap_df_cls["Cluster"] = kmeans_cls.fit_predict(shap_df_cls)
plot_tsne_clusters(shap_df_cls, cluster_labels=shap_df_cls["Cluster"], title="t-SNE Clusters (Classification)")

# Rolling Metrics for Classification
rolling_acc, rolling_f1 = calculate_rolling_metrics(y_bin, logistic_preds, window=10)
plt.plot(rolling_acc, label="Rolling Accuracy")
plt.plot(rolling_f1, label="Rolling F1-Score", color="orange")
plt.title("Rolling Metrics (Window=10)")
plt.xlabel("Time")
plt.ylabel("Score")
plt.legend()
plt.show()

# Error Trend Visualization for Key Models
kalman_errors = y_cont - kalman_predictions
plot_error_trend(kalman_errors, model_name="Kalman Filter", color="orange")

# Information Gain Calculations
info_gain_reg = calculate_information_gain(y_cont, pd.Series(kalman_predictions))
info_gain_cls = calculate_information_gain(y_bin, pd.Series(logistic_preds))
print(f"Information Gain (Regression): {info_gain_reg}")
print(f"Information Gain (Classification): {info_gain_cls}")
