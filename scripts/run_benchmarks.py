import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add `src/` to Python path
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print("Current sys.path:", sys.path)

# Now import modules from src
import yaml
import pandas as pd
from preprocessing.data_loader import load_dataset
from preprocessing.missing_value_handler import handle_missing_values
from preprocessing.encoding import encode_categorical_features


def load_config(config_path="configs/benchmark_config.yaml"):
    """
    Loads the YAML configuration file.

    Parameters:
    - config_path (str): Path to the configuration YAML file.

    Returns:
    - config (dict): Dictionary containing configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def run_benchmarks():
    # Load configuration
    config = load_config()

    # Dataset parameters
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["dataset"]["path"]))
    separator = config["dataset"]["separator"]
    target_column = config["dataset"]["target_column"]

    # Preprocessing parameters
    handle_missing = config["preprocessing"]["handle_missing_values"]
    encoding_type = config["preprocessing"]["encoding_type"]

    # Load dataset
    original_data, dataset_name = load_dataset(dataset_path, separator)

    # Handle missing values
    if handle_missing:
        cleaned_data = handle_missing_values(original_data, strategy=handle_missing)
    else:
        cleaned_data = original_data

    # Encode categorical features
    if encoding_type:
        encoded_data = encode_categorical_features(cleaned_data, target_column)
    else:
        encoded_data = cleaned_data

    print("\nPreprocessing completed.")

if __name__ == "__main__":
    run_benchmarks()
