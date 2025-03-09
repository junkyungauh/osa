import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def plot_regression_performance(metrics_dict):
    """Plot regression model performance metrics."""
    df = pd.DataFrame(metrics_dict).T
    df.plot(kind="bar", figsize=(10, 6))
    plt.title("Regression Model Performance")
    plt.ylabel("Error")
    plt.show()

def plot_classification_performance(metrics_dict):
    """Plot classification model performance metrics."""
    df = pd.DataFrame(metrics_dict).T
    df.plot(kind="bar", figsize=(10, 6))
    plt.title("Classification Model Performance")
    plt.ylabel("Metric Value")
    plt.show()

def plot_shap_importance(shap_values, feature_names):
    """Visualize SHAP importance values."""
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.mean().sort_values().plot(kind="barh", figsize=(10, 6))
    plt.title("SHAP Feature Importance")
    plt.xlabel("Mean SHAP Value")
    plt.show()

def plot_error_trend(error_series, model_name="Model", color="blue"):
    """
    Plots the trend of prediction errors over time.

    Args:
        error_series (array-like): Error values to plot.
        model_name (str): Name of the model to include in the title.
        color (str): Color for the plot line.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(error_series, color=color, label=f'{model_name} Errors')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"{model_name} Prediction Error Trend")
    plt.xlabel("Time Step")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tsne_clusters(shap_df, cluster_labels, title="t-SNE Clusters"):
    """
    Visualizes clusters using t-SNE.

    Args:
        shap_df (pd.DataFrame): SHAP values with features as columns.
        cluster_labels (array-like): Cluster labels for each data point.
        title (str): Plot title.

    Returns:
        None
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(shap_df.drop(columns='Cluster', errors='ignore'))

    tsne_df = pd.DataFrame(tsne_components, columns=['Dim1', 'Dim2'])
    tsne_df['Cluster'] = cluster_labels

    plt.figure(figsize=(10, 8))
    for cluster in tsne_df['Cluster'].unique():
        cluster_data = tsne_df[tsne_df['Cluster'] == cluster]
        plt.scatter(cluster_data['Dim1'], cluster_data['Dim2'], label=f'Cluster {cluster}', alpha=0.6)

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show()

def plot_elbow_method(shap_df, max_clusters=10, title="Elbow Method for Optimal Clusters"):
    """
    Plots the Elbow Method to determine the optimal number of clusters.

    Args:
        shap_df (pd.DataFrame): DataFrame containing SHAP values or similar data.
        max_clusters (int): Maximum number of clusters to consider.
        title (str): Plot title.

    Returns:
        None
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(shap_df)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title(title)
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()
