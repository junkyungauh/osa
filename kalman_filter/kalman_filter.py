import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, accuracy_score
from joblib import Parallel, delayed

class EnsembleKalmanFilter:
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray, P: np.ndarray, x: np.ndarray):
        """
        Initialize the Ensemble Kalman Filter.

        Args:
            F (np.ndarray): State transition matrix.
            H (np.ndarray): Observation matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Observation noise covariance matrix.
            P (np.ndarray): Initial covariance estimate.
            x (np.ndarray): Initial state estimate.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        """
        Predict the next state.
        """
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z: np.ndarray):
        """
        Update the state with a new measurement.

        Args:
            z (np.ndarray): Measurement vector (e.g., model predictions).
        """
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.F.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def filter(self, predictions: np.ndarray):
        """
        Apply the Kalman filter to a series of predictions.
    
        Args:
            predictions (np.ndarray): Sequence of input predictions.
    
        Returns:
            np.ndarray: Filtered predictions.
        """
        filtered_predictions = []
        for pred in predictions:
            self.predict()
            self.update(np.array([pred]))
            filtered_predictions.append(self.x[0])  # Extract the primary state (position)
        return np.array(filtered_predictions).flatten()  # Ensure output is 1D

class ConstantVelocityKalmanFilter(EnsembleKalmanFilter):
    def __init__(self, initial_state: np.ndarray, Q_diag: float, R_diag: float):
        """
        Kalman filter assuming constant velocity model.

        Args:
            initial_state (np.ndarray): Initial [position, velocity].
            Q_diag (float): Diagonal value of the process noise covariance matrix.
            R_diag (float): Diagonal value of the measurement noise covariance matrix.
        """
        F = np.array([[1, 1], [0, 1]])  # State transition matrix
        H = np.array([[1, 0]])          # Observation matrix
        Q = Q_diag * np.eye(2)          # Process noise covariance
        R = np.array([[R_diag]])        # Measurement noise covariance
        P = np.eye(2)                   # Initial covariance estimate
        super().__init__(F, H, Q, R, P, initial_state)


class FinancialModelKalmanFilter(EnsembleKalmanFilter):
    def __init__(self, initial_state: np.ndarray, Q_diag: float, R_diag: float, alpha: float = 0.0, beta: float = 1.0):
        """
        Kalman filter for financial time-series data incorporating alpha (drift) and beta (volatility).

        Args:
            initial_state (np.ndarray): Initial state [price].
            Q_diag (float): Diagonal value of the process noise covariance matrix.
            R_diag (float): Diagonal value of the measurement noise covariance matrix.
            alpha (float): Drift term representing expected change in the price (trend).
            beta (float): Volatility term influencing the randomness in price movement.
        """
        F = np.array([[1 + alpha]])       # Drift term added to state update
        H = np.array([[1]])              # Observation matrix
        Q = np.array([[Q_diag * beta]])  # Process noise covariance
        R = np.array([[R_diag]])         # Measurement noise covariance
        P = np.array([[1]])              # Initial covariance estimate
        super().__init__(F, H, Q, R, P, initial_state)




def optimize_kalman_hyperparameters(
    kalman_filter_creator, param_grid: list, predictions: np.ndarray, true_values: np.ndarray,
    metric: str = "mse", n_jobs: int = 1
):
    """
    Optimize hyperparameters for a given Kalman filter using grid search.
    Supports MSE, accuracy, or AUC as optimization metrics.
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

    def evaluate_params(params):
        kalman_filter = kalman_filter_creator(**params)
        kalman_predictions = kalman_filter.filter(predictions)

        # Align predictions and true values
        kalman_predictions = kalman_predictions[:len(true_values)]  # Trim predictions if necessary
        aligned_true_values = true_values[:len(kalman_predictions)]  # Trim true values if necessary

        if metric == "mse":
            score = mean_squared_error(aligned_true_values, kalman_predictions)
        elif metric == "accuracy":
            binary_predictions = (kalman_predictions > 0.5).astype(int)
            score = accuracy_score(aligned_true_values, binary_predictions)
        elif metric == "auc":
            score = roc_auc_score(aligned_true_values, kalman_predictions)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return score, params, kalman_predictions

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params) for params in ParameterGrid(param_grid)
    )

    # Select the best result based on the metric
    if metric in {"mse"}:
        best_result = min(results, key=lambda x: x[0])  # Minimize MSE
    elif metric in {"accuracy", "auc"}:
        best_result = max(results, key=lambda x: x[0])  # Maximize accuracy or AUC
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    best_score, best_params, best_predictions = best_result
    return best_params, best_predictions
