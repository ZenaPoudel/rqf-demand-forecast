import argparse
import joblib
import matplotlib.pyplot as plt
from utils import load_data


def eval_multiple_quantiles(data_filepath, trained_pipeline_filepath='./weights/trained_pipeline.pkl', quantiles=[0.05, 0.25, 0.5, 0.75, 0.9]):
    _, _, X_test, y_test = load_data(data_filepath)

    pipeline = joblib.load(trained_pipeline_filepath)

    plt.figure(figsize=(10, 6))
    for quantile in quantiles:
        y_pred = pipeline.predict(X_test, quantiles=quantile)
        plt.plot(y_test.index, y_pred, label=f'Quantile {quantile}')

    plt.plot(y_test.index, y_test.values, label='Actual', linestyle='--', color='black')
    plt.xlabel('Data Point Index')
    plt.ylabel('Demand')
    plt.title('Predictions for Different Quantiles')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot actual vs predicted curves across quantiles")
    parser.add_argument('--data_filepath', type=str, default='./demand_data.csv', help='Path to the input data file (CSV format).')

    args = parser.parse_args()

    eval_multiple_quantiles(data_filepath=args.data_filepath)
