import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Linear Regression Wrapper
class LinearRegressionWrapper:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# 2. Logistic Regression Wrapper
class LogisticRegressionWrapper:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# 3. Lasso Regression Wrapper
class LassoWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        """
        Wrapper for Lasso regression with StandardScaler to ensure compatibility with scikit-learn's GridSearchCV.
        """
        self.alpha = alpha
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(alpha=self.alpha))
        ])
        self.feature_names_ = None

    def fit(self, X, y):
        """
        Fit the pipeline (scaler + Lasso) to the training data.
        Retain feature names for consistency.
        """
        self.feature_names_ = X.columns if hasattr(X, "columns") else None
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the fitted pipeline.
        Align features if feature names are available.
        """
        if self.feature_names_ is not None:
            X = X[self.feature_names_]
        return self.pipeline.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for the pipeline.
        """
        return {"alpha": self.alpha}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        self.alpha = params.get("alpha", self.alpha)
        self.pipeline.set_params(lasso__alpha=self.alpha)
        return self


# 4. ARIMA Wrapper
# Wrapper for ARIMA
class ARIMAWrapper:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, X, y):
        # ARIMA is univariate, so only fit the target (y)
        self.model = ARIMA(y, order=self.order)
        self.fitted_model = self.model.fit()

    def predict(self, X):
        if self.fitted_model is None:
            raise ValueError("ARIMA model is not fitted.")
        # Predict for the range of the target length
        return self.fitted_model.forecast(steps=len(X))



def optimize_arima(y_train, p_values, d_values, q_values, n_jobs=1):
    """Optimize ARIMA model with grid search."""
    best_model = None
    best_score = float("inf")
    best_params = None

    # Loop through all parameter combinations
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(y_train, order=(p, d, q)).fit()
                    score = model.aic  # Using AIC as the score
                    if score < best_score:
                        best_model = model
                        best_score = score
                        best_params = (p, d, q)
                except Exception as e:
                    # Log or print error for debugging
                    print(f"ARIMA (p={p}, d={d}, q={q}) failed: {e}")

    print(f"Best ARIMA Model: order={best_params}, AIC={best_score}")
    return best_model


# 5. LSTM Model for PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize an LSTM model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Number of output features.
            num_layers (int): Number of stacked LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        out, _ = self.lstm(x)  # Get the output and hidden state
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out

    def predict(self, X):
        """
        Predict method for the LSTM model.
        Args:
            X (numpy.ndarray): Input data of shape (batch_size, seq_length, input_size).
        Returns:
            numpy.ndarray: Predicted values of shape (batch_size, output_size).
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            predictions = self.forward(inputs)
        return predictions.squeeze().numpy()  # Return 1D array for compatibility



# Wrapper for LSTM
class LSTMWrapper:
    def __init__(self, input_size, hidden_size=50, num_layers=1, num_epochs=10, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X_train, y_train):
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            outputs = self.model(torch.tensor(X_train, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        return self.model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()

    def get_params(self, deep=True):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
        }

    def set_params(self, **params):
        self.hidden_size = params.get("hidden_size", self.hidden_size)
        self.num_layers = params.get("num_layers", self.num_layers)
        self.num_epochs = params.get("num_epochs", self.num_epochs)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        return self


def optimize_lstm(X_train, y_train, input_size, hidden_sizes, learning_rates, num_epochs_list, num_layers_list=[1], n_jobs=1):
    """
    Optimize hyperparameters for an LSTM model with optional parallel processing.
    
    Args:
        X_train (numpy.ndarray): Training input data (reshaped for LSTM).
        y_train (numpy.ndarray): Training target data.
        input_size (int): Number of input features.
        hidden_sizes (list): List of hidden layer sizes to try.
        learning_rates (list): List of learning rates to try.
        num_epochs_list (list): List of epoch counts to try.
        num_layers_list (list): List of number of layers to try.
        n_jobs (int): Number of parallel jobs for optimization (default: 1).
        
    Returns:
        LSTMModel: The best LSTM model based on validation performance.
    """
    criterion = nn.MSELoss()

    def train_lstm(hidden_size, lr, num_epochs, num_layers):
        model = LSTMModel(input_size, hidden_size, 1, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            inputs = torch.tensor(X_train, dtype=torch.float32)
            targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model, loss.item(), {"hidden_size": hidden_size, "lr": lr, "num_epochs": num_epochs, "num_layers": num_layers}

    # Generate all hyperparameter combinations
    param_combinations = [
        (hidden_size, lr, num_epochs, num_layers)
        for hidden_size in hidden_sizes
        for lr in learning_rates
        for num_epochs in num_epochs_list
        for num_layers in num_layers_list
    ]

    if n_jobs > 1:
        # Parallel execution
        results = Parallel(n_jobs=n_jobs)(
            delayed(train_lstm)(hidden_size, lr, num_epochs, num_layers)
            for hidden_size, lr, num_epochs, num_layers in param_combinations
        )
    else:
        # Sequential execution
        results = [
            train_lstm(hidden_size, lr, num_epochs, num_layers)
            for hidden_size, lr, num_epochs, num_layers in param_combinations
        ]

    # Find the best model based on the lowest loss
    best_model, best_loss, best_params = min(results, key=lambda x: x[1])
    
    print(f"Best LSTM Model - Hidden Size: {best_params['hidden_size']}, "
          f"Learning Rate: {best_params['lr']}, "
          f"Num Layers: {best_params['num_layers']}, Loss: {best_loss}")
    
    return best_model


# 6. XGBoost Wrapper
class XGBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0):
        """
        Wrapper for XGBoost regressor to ensure compatibility with scikit-learn's GridSearchCV.
        Args:
            n_estimators (int): Number of trees in the ensemble.
            max_depth (int): Maximum depth of the trees.
            learning_rate (float): Learning rate (shrinkage).
            subsample (float): Subsample ratio of the training instances.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            eval_metric="rmse"  # Explicitly set the evaluation metric
        )

    def fit(self, X, y):
        """
        Fit the XGBoost model to the training data.
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
            y (pd.Series or np.ndarray): Target variable.
        Returns:
            self: Fitted model instance.
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using the fitted XGBoost model.
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Predicted values.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects.
        Returns:
            dict: Parameters of the model.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Args:
            params (dict): Dictionary of parameters to set.
        Returns:
            self: Updated instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            eval_metric="rmse"  # Explicitly set the evaluation metric
        )
        return self


# 7. Stacking Regressor Wrapper
class StackingRegressorWrapper:
    def __init__(self, base_models, meta_model):
        """
        Initialize the stacking regressor with base models and a meta-model.
        Args:
            base_models (list): List of tuples with base model names and instances.
            meta_model (object): The meta-model instance (e.g., LassoWrapper).
        """
        self.base_models = {name: model for name, model in base_models}
        self.meta_model = meta_model

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Train the stacking ensemble by first training base models,
        and then using their predictions as inputs for the meta-model.
        Args:
            X_train (np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training target.
            X_val (np.ndarray): Validation features for meta-model training.
            y_val (pd.Series or np.ndarray): Validation target for meta-model training.
        """
        # Fit base models
        for name, model in self.base_models.items():
            print(f"Fitting model: {name}")
            model.fit(X_train, y_train)

        # Generate meta-features for the meta-model
        print("Generating meta-features...")
        self.meta_features_train = np.column_stack([
            model.predict(X_val) for model in self.base_models.values()
        ])

        # Fit the meta-model using validation data
        print("Fitting meta-model...")
        self.meta_model.fit(self.meta_features_train, y_val)

    def predict(self, X_test):
        """
        Generate predictions using the stacking ensemble.
        Args:
            X_test (np.ndarray): Test features.
        Returns:
            np.ndarray: Predictions from the stacking ensemble.
        """
        # Generate meta-features for the test set
        print("Generating meta-features for test set...")
        meta_features_test = np.column_stack([
            model.predict(X_test) for model in self.base_models.values()
        ])

        # Predict using the meta-model
        print("Predicting using meta-model...")
        return self.meta_model.predict(meta_features_test)



# 8. Weighted Ensemble Utility
def ensemble_weighted_average(predictions, weights):
    """
    Perform a weighted average ensemble.

    Args:
        predictions (list of np.array): A list of model predictions.
        weights (list of float): Corresponding weights for each model.

    Returns:
        np.array: Weighted ensemble predictions.
    """
    weighted_sum = sum(w * pred for w, pred in zip(weights, predictions))
    return weighted_sum / sum(weights)

# 9. Hyperparameter Optimization Utility
def optimize_model_hyperparameters(
    model_class, param_grid, X_train, y_train, validation_data=None, stacking_features=None, scoring="neg_mean_squared_error", n_jobs=-1
):
    """
    Optimize hyperparameters for a model using GridSearchCV.
    
    Parameters:
    - model_class: The model wrapper class to be optimized.
    - param_grid: Grid of parameters to search.
    - X_train: Training features.
    - y_train: Training targets.
    - validation_data: Tuple of validation features and targets (X_val, y_val).
    - stacking_features: Additional features for meta-models (default: None).
    - scoring: Scoring metric for optimization (default: neg_mean_squared_error).
    - n_jobs: Number of parallel jobs for GridSearchCV (default: -1 for all available cores).
    
    Returns:
    - best_model: The model with the best hyperparameters.
    - best_params: The best hyperparameters found.
    """
    if stacking_features is not None:
        # Use stacking features instead of X_train if provided
        X_train = stacking_features

    model = model_class()  # Instantiate the model wrapper
    grid_search = GridSearchCV(
        model, param_grid, scoring=scoring, cv=3, verbose=1, n_jobs=n_jobs
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_



# 10. Simple Neural Network for Classification
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleNNWrapper:
    def __init__(self, input_size, output_size, num_epochs=10, learning_rate=0.01):
        self.model = SimpleNN(input_size, output_size)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        for epoch in range(self.num_epochs):
            self.model.train()
            outputs = self.model(torch.tensor(X, dtype=torch.float32))
            loss = self.criterion(outputs, torch.tensor(y, dtype=torch.long))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()


class RandomForestWrapper(BaseEstimator):
    """
    Wrapper for RandomForest to integrate with hyperparameter optimization and model evaluation.
    Supports both regression and classification tasks.
    """
    def __init__(self, task_type='regression', **kwargs):
        self.task_type = task_type
        if task_type == 'regression':
            self.model = RandomForestRegressor(**kwargs)
        elif task_type == 'classification':
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("task_type must be either 'regression' or 'classification'")

    def fit(self, X, y):
        """Fits the model to the data."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Generates predictions using the fitted model."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Generates prediction probabilities for classification tasks."""
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        raise AttributeError("predict_proba is only available for classification tasks")

    def get_params(self, deep=True):
        """Returns the parameters of the model."""
        return self.model.get_params(deep)

    def set_params(self, **params):
        """Sets the parameters of the model."""
        self.model.set_params(**params)
        return self



# PyTorch Neural Network for Ensemble
class NeuralNetworkEnsemble(nn.Module):
    def __init__(self, input_size):
        """
        A single-layer neural network for ensemble learning.
        Args:
            input_size (int): Number of base model predictions (features) used as input.
        """
        super(NeuralNetworkEnsemble, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

class NeuralNetworkEnsembleWrapper:
    def __init__(self, input_size, hidden_size=50, learning_rate=0.01, num_epochs=100):
        """
        Initialize the neural network ensemble wrapper with customizable parameters.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in the layer.
            learning_rate (float): Learning rate for optimizer.
            num_epochs (int): Number of epochs for training.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Define the neural network model
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X_train, y_train):
        """
        Train the neural network on the provided training data.
        Args:
            X_train (np.ndarray): Input training features.
            y_train (np.ndarray or pd.Series): Input training targets.
        """
        self.model.train()
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values if isinstance(y_train, pd.Series) else y_train, dtype=torch.float32).view(-1, 1)
    
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()


    def predict(self, X_test):
        """
        Generate predictions using the trained neural network.
        Args:
            X_test (np.ndarray): Input test features.
        Returns:
            np.ndarray: Predicted values.
        """
        self.model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_test).squeeze().numpy()
        return predictions
