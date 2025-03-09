from preprocessing.preprocessing import normalize_data, create_lagged_features
from models.models import linear_regression_model, lasso_regression_model, stacking_regression_model
from metrics.metrics import calculate_regression_metrics, calculate_regression_information_gain
from visualizations.visualizations import plot_error_trend, plot_heatmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
cont_data_path = 'simulated_series_cont.csv'

# Load Continuous Data
cont_data = pd.read_csv(cont_data_path)
cont_data = create_lagged_features(cont_data, column="return", lag=1)
cont_data = normalize_data(cont_data, method="z-score")
X_cont = cont_data.drop(columns=["return"])
y_cont = cont_data["return"]

# Models
lasso_model = lasso_regression_model(X_cont, y_cont, alpha=0.5)
lasso_predictions = lasso_model.predict(X_cont)

stacking_model = stacking_regression_model(
    base_models=[
        ("linear", linear_regression_model(X_cont, y_cont)),
        ("lasso", lasso_model),
    ],
    meta_model=linear_regression_model(X_cont, y_cont),
    X_train=X_cont,
    y_train=y_cont
)
stacking_predictions = stacking_model.predict(X_cont)

# Calculate Errors
lasso_residuals = y_cont - lasso_predictions
stacking_residuals = y_cont - stacking_predictions

# Error Metrics
lasso_metrics = calculate_regression_metrics(y_cont, lasso_predictions)
stacking_metrics = calculate_regression_metrics(y_cont, stacking_predictions)

# Print Metrics
print("Lasso Regression Metrics:", lasso_metrics)
print("Stacking Regression Metrics:", stacking_metrics)

# Plot Error Trends
plt.figure(figsize=(10, 5))
plt.plot(lasso_residuals, label="Lasso Residuals", alpha=0.7)
plt.plot(stacking_residuals, label="Stacking Residuals", alpha=0.7)
plt.title("Residual Trend Comparison")
plt.legend()
plt.show()

# Error Heatmap
errors_df = pd.DataFrame({
    "Lasso": lasso_residuals,
    "Stacking": stacking_residuals
})
plot_heatmap(errors_df.corr(), title="Residual Correlation Heatmap")

# Error Distribution Comparison
plt.figure(figsize=(10, 5))
plt.hist(lasso_residuals, bins=20, alpha=0.5, label="Lasso Residuals")
plt.hist(stacking_residuals, bins=20, alpha=0.5, label="Stacking Residuals")
plt.title("Residual Distribution Comparison")
plt.legend()
plt.show()
