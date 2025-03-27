import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MovingAverageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.learned_values = {}

    def fit(self, X, y=None):
        df = X.copy()
        df['demand'] = y

        # Sort data by Date
        df = df.sort_values(by=['date'])

        # Compute moving averages
        df['avg_1week'] = df['demand'].rolling(window=7, min_periods=1).mean()
        df['avg_1month'] = df['demand'].rolling(window=30, min_periods=1).mean()
        df['avg_1year'] = df['demand'].rolling(window=365, min_periods=1).mean()

        # Interpolate NaN values after computing the moving averages
        df['avg_1week'] = df['avg_1week'].interpolate(method='linear')
        df['avg_1month'] = df['avg_1month'].interpolate(method='linear')
        df['avg_1year'] = df['avg_1year'].interpolate(method='linear')

        # Store the last values for each date
        self.learned_values = df.tail(1)[['avg_1week', 'avg_1month', 'avg_1year']].to_dict('index')

        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Create empty columns for moving averages
        X['avg_1week'] = np.nan
        X['avg_1month'] = np.nan
        X['avg_1year'] = np.nan

        # Fill moving averages using the learned values
        for _, values in self.learned_values.items():
            X['avg_1week'] = values['avg_1week']
            X['avg_1month'] = values['avg_1month']
            X['avg_1year'] = values['avg_1year']

        return X


class TimeFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['date'] = pd.to_datetime(X['date'])
        X['day_of_year'] = X['date'].dt.dayofyear
        X['day_of_month'] = X['date'].dt.day
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.dayofweek
        return X
    

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]