# Demand Forecasting with Quantile Regression Forest

<br />

# Data

[Sales Demand Dataset](https://www.kaggle.com/datasets/sunilgautam/sales-demand-dataset)

<br />

# Usage Guide

## Install Dependencies

1. **Install Python:** Make sure Python is installed on your system. If not, you can download and install Python from the official Python website: https://www.python.org/downloads/

2. **Create a virtual environment:** 

	```bash
	python -m venv env
	```

3. **Activate the virtual environment**

	> For Windows CMD Users 
	```bash
	.\env\Scripts\Activate.bat
 	```
 
	> For Windows Powershell Users 
	```bash
	.\env\Scripts\Activate.ps1
	``` 

	> For macOS/Linux Users
	```bash
	source env/bin/activate
	```

4. **Install the dependencies**
	
	```bash
	pip install -r requirements.txt
	```

<br />


## Download Dataset from KaggleHub

```bash
python download_data.py
```

<br />

## Training

```bash
python train.py --data_filepath DATA_PATH [--save_filepath SAVE_PATH] [--random_seed RANDOM_SEED]
```

#### Flags:

- **--data_filepath:** Specifies the file path to the dataset. By default, it points to ./demand_data.csv.

- **--save_filepath:** The file name where the trained model will be saved. By default, it saves as trained_pipeline.pkl in the ./weights/ directory.

- **--random_seed:** Random seed used for model reproducibility. By default, it is set to 42.



<br />

## Evaluation

```bash
python eval.py --data_filepath DATA_PATH [--quantile QUANTILE] [--trained_pipeline_filepath TRAINED_PIPELINE_PATH]
```

#### Flags:

- **--data_filepath:** Specifies the file path to the dataset. By default, it points to `./demand_data.csv`.
- **--quantile:** Specifies the quantile for which the prediction is to be evaluated (float value). By default, it is set to `0.5`.
- **--trained_pipeline_filepath:** The file path to the trained model pipeline. By default, it points to `./weights/trained_pipeline.pkl`.

