import os
import pandas as pd
import yaml
from preprocessing.data_loader import load_dataset
from preprocessing.missing_value_handler import handle_missing_values
from preprocessing.encoding import encode_categorical_features

# Load configuration
def load_config(config_path="configs/benchmark_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_preprocessing(dataset_path, separator, target_column):
    """
    Preprocesses the dataset (handling missing values, encoding) and returns it.

    Returns:
    - encoded_data (DataFrame): The cleaned and encoded dataset.
    - dataset_name (str): Name of the dataset.
    """
    config = load_config()
    handle_missing = config["preprocessing"]["handle_missing_values"]
    encoding_type = config["preprocessing"]["encoding_type"]

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    cleaned_dataset_path = f"datasets/cleaned/{dataset_name}_cleaned.csv"

    # Check if cleaned dataset already exists
    if os.path.exists(cleaned_dataset_path):
        print(f"✅ Cleaned dataset found: {cleaned_dataset_path}. Skipping preprocessing.")
        return pd.read_csv(cleaned_dataset_path), dataset_name  # Return DataFrame directly

    # Load dataset
    original_data, dataset_name = load_dataset(dataset_path, separator)

    # Handle missing values
    cleaned_data = handle_missing_values(original_data, strategy=handle_missing)

    # Encode categorical features
    if encoding_type:
        encoded_data = encode_categorical_features(cleaned_data, target_column)
    else:
        encoded_data = cleaned_data

    print(f" Preprocessing completed. Saving cleaned dataset to {cleaned_dataset_path}...")
    os.makedirs("datasets/cleaned", exist_ok=True)  # Ensure folder exists
    encoded_data.to_csv(cleaned_dataset_path, index=False)

    return encoded_data, dataset_name  # Return DataFrame directly instead of path

if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset"]["path"]
    separator = config["dataset"]["separator"]
    target_column = config["dataset"]["target_column"]

    run_preprocessing(dataset_path, separator, target_column)
