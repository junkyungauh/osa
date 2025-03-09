import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from models.models import LassoWrapper, optimize_model_hyperparameters


from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100 if np.any(y_true != 0) else np.nan
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}



# Debugging Setup: Replace 'X_cont' and 'y_cont' with your dataset variables
# Load your dataset or define X_cont and y_cont before running

# 1. Replace Features with Random Noise
X_random = pd.DataFrame(np.random.randn(*X_cont.shape), columns=X_cont.columns)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_random, y_cont, test_size=0.3, random_state=42)

# Train and evaluate Lasso with random features
lasso_random, _ = optimize_model_hyperparameters(
    LassoWrapper, {"alpha": np.logspace(-8, 2, 50)}, X_train_r, y_train_r, validation_data=(X_test_r, y_test_r)
)
lasso_random_preds = lasso_random.predict(X_test_r)
random_metrics = calculate_regression_metrics(y_test_r, lasso_random_preds)
print("Lasso Metrics with Random Features:", random_metrics)

# 2. Check Correlations Between Random Features and Target
correlations = X_random.corrwith(y_cont)
print("Correlations with Random Features:", correlations)

# 3. Shuffle Target and Evaluate Model
y_shuffled = shuffle(y_cont, random_state=42).reset_index(drop=True)
lasso_random_shuffled, _ = optimize_model_hyperparameters(
    LassoWrapper, {"alpha": np.logspace(-8, 2, 50)}, X_train_r, y_shuffled.iloc[:len(X_train_r)],
    validation_data=(X_test_r, y_shuffled.iloc[len(X_train_r):])
)
lasso_random_shuffled_preds = lasso_random_shuffled.predict(X_test_r)
shuffled_metrics = calculate_regression_metrics(y_shuffled.iloc[len(X_train_r):], lasso_random_shuffled_preds)
print("Lasso Metrics with Shuffled Target:", shuffled_metrics)

# 4. Overlap Check Between Splits
print("Overlap between Train and Test sets:", X_train_r.index.intersection(X_test_r.index))

# 5. Evaluate Linear Regression with Random Features
lr = LinearRegression()
lr.fit(X_train_r, y_train_r)
lr_preds = lr.predict(X_test_r)
lr_metrics = calculate_regression_metrics(y_test_r, lr_preds)
print("Linear Regression Metrics with Random Features:", lr_metrics)

# 6. Visualize Target Distribution
sns.histplot(y_cont, bins=50, kde=True)
plt.title("Target Variable Distribution")
plt.show()

# 7. Add Noise to Target and Evaluate Lasso
noise_std = y_cont.std() * 0.5
y_noisy = y_cont + np.random.normal(0, noise_std, len(y_cont))
lasso_noisy, _ = optimize_model_hyperparameters(
    LassoWrapper, {"alpha": np.logspace(-8, 2, 50)}, X_train_r, y_noisy.iloc[:len(X_train_r)],
    validation_data=(X_test_r, y_noisy.iloc[len(X_train_r):])
)
lasso_noisy_preds = lasso_noisy.predict(X_test_r)
noisy_metrics = calculate_regression_metrics(y_noisy.iloc[len(X_train_r):], lasso_noisy_preds)
print("Lasso Metrics with Noisy Target:", noisy_metrics)

# 8. Permuted Cross-Validation with Lasso
permuted_mae = []
for _ in range(50):
    y_perm = shuffle(y_cont)
    lasso_perm, _ = optimize_model_hyperparameters(
        LassoWrapper, {"alpha": np.logspace(-8, 2, 50)}, X_train_r, y_perm.iloc[:len(X_train_r)],
        validation_data=(X_test_r, y_perm.iloc[len(X_train_r):])
    )
    permuted_preds = lasso_perm.predict(X_test_r)
    permuted_mae.append(mean_absolute_error(y_perm.iloc[len(X_train_r):], permuted_preds))
print("Mean Permuted MAE:", np.mean(permuted_mae))

# 9. Cross-Validation MAE on Random Features
cv_mae = cross_val_score(LassoWrapper(alpha=0.1), X_train_r, y_train_r, scoring='neg_mean_absolute_error', cv=5)
print("Cross-Validation MAE on Random Features:", -np.mean(cv_mae))

# 10. Regularization Analysis
alphas = np.logspace(-8, 2, 50)
maes = []
for alpha in alphas:
    lasso = LassoWrapper(alpha=alpha)
    lasso.fit(X_train_r, y_train_r)
    preds = lasso.predict(X_test_r)
    maes.append(mean_absolute_error(y_test_r, preds))
plt.plot(alphas, maes)
plt.xscale('log')
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("MAE")
plt.title("Effect of Regularization Strength on MAE")
plt.show()

# 11. Residual Analysis for Best Model
residuals = y_test_r - lasso_noisy.predict(X_test_r)
plt.scatter(lasso_noisy.predict(X_test_r), residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predictions")
plt.show()

# 12. Minimal Preprocessing for Simplicity
def preprocess_minimal_features(file_path, target_column, lag_steps=None, normalize=True):
    data = pd.read_csv(file_path)
    feature_data = pd.DataFrame(index=data.index)
    if lag_steps:
        for lag in lag_steps:
            feature_data[f'{target_column}_lag{lag}'] = data[target_column].shift(lag)
    feature_data.dropna(inplace=True)
    data = data.loc[feature_data.index]
    X = feature_data
    y = data.loc[feature_data.index, target_column]
    if normalize:
        X = (X - X.mean()) / X.std()  # Simple normalization
    return X, y

# Preprocess Minimal Features
X_minimal, y_minimal = preprocess_minimal_features(cont_data_path, target_column="return", lag_steps=[1, 2, 3])
X_train_min, X_test_min, y_train_min, y_test_min = train_test_split(
    X_minimal, y_minimal, test_size=0.2, random_state=42, shuffle=False
)

# Train Lasso on Minimal Features
lasso_min, _ = optimize_model_hyperparameters(
    LassoWrapper, {"alpha": np.logspace(-8, 2, 50)}, X_train_min, y_train_min,
    validation_data=(X_test_min, y_test_min)
)
lasso_min_preds = lasso_min.predict(X_test_min)
min_metrics = calculate_regression_metrics(y_test_min, lasso_min_preds)
print("Lasso Metrics with Minimal Features:", min_metrics)
