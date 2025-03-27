import random
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from typing import Tuple, Any


def reset_random(random_seed: int = 42) -> None:
    # Set random seed for Python's built-in random module
    random.seed(random_seed)

    # Set random seed for NumPy
    np.random.seed(random_seed)


def load_data(data_path: str, test_days: int = 60) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Split data into train and test (last N days)
    cutoff_date = df['date'].max() - pd.Timedelta(days=test_days)
    train = df[df['date'] < cutoff_date]
    test = df[df['date'] >= cutoff_date]

    X_train, y_train = train.drop(columns='demand'), train['demand']
    X_test, y_test = test.drop(columns='demand'), test['demand']

    return X_train, y_train, X_test, y_test


def mean_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true - y_pred)


def mean_absolute_error_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Exclude zero actuals to avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def weighted_mape(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return np.sum(weights * np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) / np.sum(weights) * 100


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, quantile: Any = None) -> None:
    y_pred = pipeline.predict(X_test, quantiles=quantile)

    r2_score_value = r2_score(y_test, y_pred)
    me = mean_error(y_test, y_pred)
    mae = mean_absolute_error_custom(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Weighted errors
    weights = y_test  # Using true values as weights
    weighted_mae_value = weighted_mae(y_test, y_pred, weights)
    weighted_mape_value = weighted_mape(y_test, y_pred, weights)
    
    # Print all metrics
    print(f"Model R^2 Score: {r2_score_value:.2f}")
    print(f"Mean Error (ME): {me:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Weighted Mean Absolute Error (Weighted MAE): {weighted_mae_value:.2f}")
    print(f"Weighted Mean Absolute Percentage Error (Weighted MAPE): {weighted_mape_value:.2f}%")