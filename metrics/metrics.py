from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, log_loss
import numpy as np

def calculate_regression_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    # Safeguard against division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.nan
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def calculate_classification_metrics(y_true, y_pred):
    """Calculate common classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"Accuracy": accuracy, "F1 Score": f1}

def calculate_regression_information_gain(y_true, y_pred):
    """Calculate information gain for regression models based on variance reduction."""
    baseline_variance = np.var(y_true)
    residual_variance = np.var(y_true - y_pred)
    info_gain = baseline_variance - residual_variance
    return {
        "Baseline Variance": baseline_variance,
        "Residual Variance": residual_variance,
        "Information Gain": info_gain
    }

def calculate_classification_information_gain(y_true, y_pred_proba):
    """Calculate information gain for classification models based on entropy reduction."""
    # Baseline entropy (entropy of class distribution)
    class_prob = np.bincount(y_true) / len(y_true)
    baseline_entropy = -np.sum(class_prob * np.log(class_prob + 1e-10))  # Avoid log(0)

    # Post-model entropy (log loss)
    post_entropy = log_loss(y_true, y_pred_proba, normalize=True)

    # Information gain
    info_gain = baseline_entropy - post_entropy
    return {
        "Baseline Entropy": baseline_entropy,
        "Post-Model Entropy": post_entropy,
        "Information Gain": info_gain
    }
