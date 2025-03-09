from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

def load_data(file_path, index_col=None):
    """Load data from a CSV file."""
    return pd.read_csv(file_path, index_col=index_col)

from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rolling_continuous_metrics(y_true, y_pred, window=10):
    """
    Calculates rolling MSE and MAE for regression tasks.

    Args:
        y_true (array-like): True continuous values.
        y_pred (array-like): Predicted continuous values.
        window (int): Rolling window size.

    Returns:
        tuple: Rolling MSE and MAE as Pandas Series.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    rolling_mse = df["y_true"].rolling(window).apply(
        lambda x: mean_squared_error(x, df["y_pred"].iloc[x.index]), raw=False
    )
    rolling_mae = df["y_true"].rolling(window).apply(
        lambda x: mean_absolute_error(x, df["y_pred"].iloc[x.index]), raw=False
    )
    return rolling_mse.dropna(), rolling_mae.dropna()


def split_data_indices(total_length, train_ratio=0.8, val_ratio=0.1):
    """Generate train, validation, and test indices based on ratios."""
    train_end = int(total_length * train_ratio)
    val_end = train_end + int(total_length * val_ratio)
    return range(train_end), range(train_end, val_end), range(val_end, total_length)



def calculate_rolling_binary_metrics(y_true, y_pred, window=10):
    """
    Calculates rolling accuracy and F1-score for binary classification.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted binary labels.
        window (int): Rolling window size.

    Returns:
        tuple: Rolling accuracy and F1-score as Pandas Series.
    """
    # Ensure inputs are aligned and converted to DataFrame
    data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).reset_index(drop=True)

    # Initialize rolling metrics
    rolling_accuracy = []
    rolling_f1 = []

    # Iterate over windows
    for i in range(len(data) - window + 1):
        window_true = data["y_true"].iloc[i:i + window]
        window_pred = data["y_pred"].iloc[i:i + window]

        acc = accuracy_score(window_true, window_pred)
        f1 = f1_score(window_true, window_pred, zero_division=0)

        rolling_accuracy.append(acc)
        rolling_f1.append(f1)

    # Align output to the original index
    rolling_accuracy = pd.Series(rolling_accuracy, index=data.index[window - 1:])
    rolling_f1 = pd.Series(rolling_f1, index=data.index[window - 1:])

    return rolling_accuracy, rolling_f1