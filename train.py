import os
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
import joblib
from quantile_forest import RandomForestQuantileRegressor
from utils import reset_random, load_data
from transformers import MovingAverageTransformer, TimeFeatureTransformer, FeatureSelector


def train(args):
    X_train, y_train, _, _ = load_data(args.data_filepath)

    features = [
        "day_of_year",
        "day_of_month",
        "month",
        "day",
        "avg_1week",
        "avg_1month",
        "avg_1year",
        "advertising_spend",
        "holiday_promotions",
        "market_sentiment"
        ]
    numerical_features = [
        "day_of_year",
        "day_of_month",
        "month",
        "day",
        "avg_1week",
        "avg_1month",
        "avg_1year",
        "advertising_spend",
        "holiday_promotions"
        ]
    categorical_features = ["market_sentiment"]

    # Pipeline
    pipeline = Pipeline([
        ('moving_avg', MovingAverageTransformer()),
        ('time_features', TimeFeatureTransformer()),
        ('feature_selector', FeatureSelector(features)),
        ('column_transformer', ColumnTransformer([
            ('num', RobustScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])),
        ('regressor', RandomForestQuantileRegressor(random_state=args.random_seed, verbose=1))
    ])

    reset_random(args.random_seed)
    print('Fitting the pipeline ...')
    pipeline.fit(X_train, y_train)

    # Save the trained pipeline
    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    joblib.dump(pipeline, os.path.join('./weights', args.save_filepath))
    print(f'Weights saved to /weights/{args.save_filepath}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Quantile Regression Forest for Demand Forecasting.")
    parser.add_argument('--data_filepath', type=str, default='./demand_data.csv', help='Path to the input data file (CSV format).')
    parser.add_argument('--save_filepath', type=str, default='trained_pipeline.pkl', help='Filename to save the trained model (with .pkl extension).')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    train(args)