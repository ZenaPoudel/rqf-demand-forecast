import os
import kagglehub
import shutil


# Download dataset
path = kagglehub.dataset_download("sunilgautam/sales-demand-dataset")

# Define the destination as the current working directory
destination_folder = os.getcwd()

# Path to the CSV file
data_file_path = os.path.join(path, 'demand_data.csv')

# Move the CSV file to the current working directory
shutil.move(data_file_path, os.path.join(destination_folder, 'demand_data.csv'))
print(f"Dataset downloaded to: {os.path.join(destination_folder, 'demand_data.csv')}")