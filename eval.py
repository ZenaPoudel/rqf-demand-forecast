import argparse
import joblib
from utils import load_data, evaluate_model


def eval(args):
    _, _, X_test, y_test = load_data(args.data_filepath)

    pipeline = joblib.load(args.trained_pipeline_filepath)

    evaluate_model(pipeline, X_test, y_test, args.quantile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the model for Demand Forecasting.")
    parser.add_argument('--data_filepath', type=str, default='./demand_data.csv', help='Path to the input data file (CSV format).')
    parser.add_argument('--quantile', type=float, default=0.5, help='Quantile (float)')
    parser.add_argument('--trained_pipeline_filepath', type=str, default='./weights/trained_pipeline.pkl', help='Path to the trained pipeline.')
    
    args = parser.parse_args()

    eval(args)