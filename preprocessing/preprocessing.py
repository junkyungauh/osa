
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def normalize_data(df, method="z-score"):
    """Normalize data using z-score or logistic transformation."""
    if method == "z-score":
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif method == "logistic":
        return 1 / (1 + np.exp(-df))
    else:
        raise ValueError("Normalization method must be 'z-score' or 'logistic'.")


def create_lagged_features(df, column, lag=1):
    """Create lagged features for a given column."""
    df[f"{column}_lag{lag}"] = df[column].shift(lag)
    return df.dropna().reset_index(drop=True)


def split_temporal_data(X, y, val_size=0.1, test_size=0.1):
    """Split features and target into train, validation, and test sets."""
    total_len = len(X)
    test_len = int(total_len * test_size)
    val_len = int(total_len * val_size)
    train_len = total_len - test_len - val_len

    X_train, X_val, X_test = X.iloc[:train_len], X.iloc[train_len:train_len + val_len], X.iloc[train_len + val_len:]
    y_train, y_val, y_test = y.iloc[:train_len], y.iloc[train_len:train_len + val_len], y.iloc[train_len + val_len:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def continuous_to_categorical(df, column, bins, labels, one_hot=False):
    """Convert a continuous column into categories and optionally one-hot encode."""
    df[f"{column}_category"] = pd.cut(df[column], bins=bins, labels=labels)
    if one_hot:
        encoder = OneHotEncoder(sparse=False)
        encoded = encoder.fit_transform(df[[f"{column}_category"]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{column}_cat_{i}" for i in range(len(labels))])
        return pd.concat([df, encoded_df], axis=1)
    return df.dropna()


# Updated Preprocessing Function
def preprocess_data_with_general_features(file_path, target_column, lag_steps=None, rolling_window=None, ema_window=None):
    data = pd.read_csv(file_path)

    # Validate data sorting
    assert data.index.is_monotonic_increasing, "Dataset is not sorted by time."
    assert rolling_window >= 3, "Rolling window size should be at least 3 to avoid capturing noise."

    feature_data = pd.DataFrame(index=data.index)
    # Feature Engineering (unchanged)
    signal_cols = [col for col in data.columns if col.startswith('sig')]
    for col in signal_cols:
        feature_data[f'{col}_roll_mean'] = data[col].rolling(window=rolling_window, closed='right').mean()
        feature_data[f'{col}_roll_std'] = data[col].rolling(window=rolling_window, closed='right').std()

    # Lagged Features
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)

    # Drop NaNs introduced by lagging
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]

    # Separate features and target
    X = feature_data
    y = data[target_column]

    return X, y

# # Load and preprocess data
# X_cont, y_cont = preprocess_data_with_general_features(
#     file_path='simulated_series_cont.csv',
#     target_column='return',
#     lag_steps=[1, 2, 3],
#     rolling_window=10,
#     ema_window=5
# )
#
# X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(
#     X_cont, y_cont, train_size=0.5, val1_size=0.15, val2_size=0.05, kalman_size=0.1, test_size=0.2
# )



# used in full_demo_bin
def preprocess_bin_data_with_features(file_path, target_column, lag_steps=None, rolling_window=None):
    """Load and preprocess data with feature engineering."""
    data = pd.read_csv(file_path)

    # Ensure data is sorted by time
    assert data.index.is_monotonic_increasing, "Dataset is not sorted by time."

    feature_data = pd.DataFrame(index=data.index)
    signal_cols = [col for col in data.columns if col.startswith('sig')]

    # Rolling features
    for col in signal_cols:
        feature_data[f'{col}_roll_mean'] = data[col].rolling(window=rolling_window).mean()
        feature_data[f'{col}_roll_std'] = data[col].rolling(window=rolling_window).std()

    # Lag features
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)

    # Drop rows with NaNs
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]

    X = feature_data
    y = data[target_column]
    return X, y

# X, y = preprocess_bin_data_with_features(
#     file_path='simulated_series_bin.csv',
#     target_column='binary',
#     lag_steps=[1, 2, 3],
#     rolling_window=10
# )
#
# X_train, X_val1, X_val2, X_kalman, X_test, y_train, y_val1, y_val2, y_kalman, y_test = five_way_split(
#     X, y, train_size=0.5, val1_size=0.15, val2_size=0.1, kalman_size=0.1, test_size=0.15
# )
