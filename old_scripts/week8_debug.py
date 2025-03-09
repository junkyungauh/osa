# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models.models import (
    LassoWrapper,
    RandomForestWrapper,
    XGBoostWrapper,
    optimize_model_hyperparameters,
)

# Helper function: Calculate regression metrics
def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}

# Visualization: Splits in the dataset
def visualize_splits(X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test):
    plt.figure(figsize=(15, 10))
    plt.plot(y_train.index, y_train, label="Train", color="blue", alpha=0.7)
    plt.plot(y_val1.index, y_val1, label="Validation 1", color="orange", alpha=0.7)
    plt.plot(y_val2.index, y_val2, label="Validation 2", color="green", alpha=0.7)
    plt.plot(y_kalman.index, y_kalman, label="Kalman", color="purple", alpha=0.7)
    plt.plot(y_test.index, y_test, label="Test", color="red", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("Target Variable (y)")
    plt.title("Target Variable Across Train, Validation, Kalman, and Test Splits")
    plt.legend()
    plt.grid(True)
    plt.show()

# Comprehensive debugging pipeline
def debug_pipeline(X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test, features=None):
    splits = {
        "Train": (X_train, y_train),
        "Validation 1": (X_val1, y_val1),
        "Validation 2": (X_val2, y_val2),
        "Kalman": (X_kalman, y_kalman),
        "Test": (X_test, y_test),
    }
    features = features or X_train.columns[:5]  # Default: first 5 features

    # Split validation
    print("Split Validation:")
    for split_name, (X, y) in splits.items():
        print(f"{split_name}: {len(X)} samples")
    print("\nChecking for overlaps...")
    print("Overlap between Train and Test sets:", X_train.index.intersection(X_test.index))
    print("Overlap between Train and Validation 1 sets:", X_train.index.intersection(X_val1.index))
    print("Overlap between Validation 1 and Validation 2 sets:", X_val1.index.intersection(X_val2.index))
    print("Overlap between Validation 2 and Test sets:", X_val2.index.intersection(X_test.index))

    # Feature validation
    print("\nFeature Validation:")
    for feature in features:
        print(f"\nFeature: {feature}")
        for split_name, (X, y) in splits.items():
            if feature in X.columns:
                print(f"{split_name} - Mean: {X[feature].mean():.4f}, Std: {X[feature].std():.4f}")
            else:
                print(f"{split_name} - Feature '{feature}' not found. Skipping.")

    # Normalization checks
    print("\nNormalization Checks (Mean and Std):")
    for split_name, (X, _) in splits.items():
        print(f"{split_name}: Mean = {X.mean().mean():.4f}, Std = {X.std().mean():.4f}")

    # Target distribution across splits
    print("\nTarget Distribution Across Splits:")
    for split_name, (_, y) in splits.items():
        print(f"{split_name} - Mean: {y.mean():.4f}, Std: {y.std():.4f}")

    # Visualize feature trends
    print("\nVisualizing Feature Trends...")
    for feature in features:
        if feature in X_train.columns:
            plt.figure(figsize=(10, 6))
            for split_name, (X, y) in splits.items():
                if feature in X.columns:
                    plt.plot(X.index, X[feature], label=f"{split_name}: {feature}", alpha=0.6)
            plt.title(f"Feature Trends: {feature}")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

    # Visualize target trends
    print("\nVisualizing Target Trends...")
    plt.figure(figsize=(10, 6))
    for split_name, (_, y) in splits.items():
        plt.plot(y.index, y, label=f"{split_name}: Target", alpha=0.6)
    plt.title("Target Trends Across Splits")
    plt.xlabel("Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.show()

    # Model overfitting checks
    print("\nModel Overfitting Checks:")
    lasso_correlation = X_train.corrwith(y_train)
    print("Correlation of Lasso-selected features with target (Train):")
    print(lasso_correlation.sort_values(ascending=False).head(10))

# Feature importance analysis
# Feature importance analysis
def feature_importance_analysis(X_train, rf_model, lasso_model):
    print("\nRandom Forest Feature Importance:")
    try:
        # Extract feature importances from the underlying model
        rf_feature_importances = rf_model.model.feature_importances_
        # Check for alignment between the number of features
        if len(rf_feature_importances) == X_train.shape[1]:
            rf_importance = pd.Series(rf_feature_importances, index=X_train.columns).sort_values(ascending=False)
            print(rf_importance.head(10))
        else:
            print("Mismatch in feature names and importances. Ensure the model was trained on the same features.")
    except AttributeError:
        print("Random Forest model does not expose 'feature_importances_' attribute.")

    print("\nLasso Coefficients:")
    try:
        if len(lasso_model.coef_) == X_train.shape[1]:
            lasso_coefficients = pd.Series(lasso_model.coef_, index=X_train.columns).sort_values(ascending=False)
            print(lasso_coefficients.head(10))
        else:
            print("Mismatch in feature names and coefficients. Ensure the model was trained on the same features.")
    except AttributeError:
        print("Lasso model does not expose 'coef_' attribute.")



# Visualize residuals
def visualize_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predictions")
    plt.xlabel("Index")
    plt.ylabel("Residuals")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Assuming data is preprocessed and split as X_train, X_val1, etc.
    visualize_splits(X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test)
    debug_pipeline(
        X_train, X_val1, X_val2, X_kalman, X_test,
        y_train, y_val1, y_val2, y_kalman, y_test,
        features=["return_lag1", "return_roll_mean_10", "sin_time"],
    )
    # Call the feature importance analysis
    feature_importance_analysis(X_train, rf_model, lasso_base_model)

    # Evaluate models again
    lasso_test_preds = lasso_base_model.predict(X_test)
    rf_test_preds = rf_model.predict(X_test)
    xgb_test_preds = xgb_model.predict(X_test)
    print("Evaluation on Test Set:")
    print("Lasso:", calculate_regression_metrics(y_test, lasso_test_preds))
    print("Random Forest:", calculate_regression_metrics(y_test, rf_test_preds))
    print("XGBoost:", calculate_regression_metrics(y_test, xgb_test_preds))

    # Visualize residuals
    visualize_residuals(y_test, lasso_test_preds)
