import os
import pandas as pd
import yaml
from preprocessing.data_loader import load_dataset
from preprocessing.missing_value_handler import handle_missing_values
from preprocessing.encoding import encode_categorical_features
from sklearn.model_selection import train_test_split

# Load configuration
def load_config(config_path="configs/benchmark_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_preprocessing(dataset_path, separator, target_column):
    """
    Preprocesses the dataset (handling missing values, encoding), splits out a test set, and saves the cleaned dataset.

    Returns:
    - cleaned_data (DataFrame): The cleaned and encoded dataset used for training.
    - dataset_name (str): The name of the dataset.
    - original_data (DataFrame): The raw dataset before preprocessing.
    - test_set (DataFrame): The portion set aside for final evaluation.
    """
    config = load_config()
    test_size = config["dataset"]["test_size"]
    handle_missing = config["preprocessing"]["handle_missing_values"]
    encoding_type = config["preprocessing"]["encoding_type"]

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    cleaned_dataset_path = f"datasets/cleaned/{dataset_name}_cleaned.csv"
    test_dataset_path = f"datasets/test/{dataset_name}_test_set.csv"  # Save test set here

    # Ensure necessary directories exist
    os.makedirs("datasets/cleaned", exist_ok=True)
    os.makedirs("datasets/test", exist_ok=True)

    # Check if datasets already exist
    if os.path.exists(cleaned_dataset_path) and os.path.exists(test_dataset_path):
        print(f"Cleaned dataset found: {cleaned_dataset_path}. Skipping preprocessing.")
        cleaned_data = pd.read_csv(cleaned_dataset_path)
        test_set = pd.read_csv(test_dataset_path)
        original_data, _ = load_dataset(dataset_path, separator)
    else:
        # Load dataset
        original_data, dataset_name = load_dataset(dataset_path, separator)

        # Handle missing values
        cleaned_data = handle_missing_values(original_data, strategy=handle_missing)

        # Encode categorical features
        if encoding_type:
            cleaned_data = encode_categorical_features(cleaned_data, target_column)

        # Split into cleaned_data (training) and test_set
        cleaned_data, test_set = train_test_split(
            cleaned_data,
            test_size=test_size,
            stratify=cleaned_data[target_column],
            random_state=42
        )

        # Save cleaned dataset and test set
        cleaned_data.to_csv(cleaned_dataset_path, index=False)
        test_set.to_csv(test_dataset_path, index=False)

        print(f"Preprocessing completed. Saved cleaned dataset to {cleaned_dataset_path}.")
        print(f"Saved test set to {test_dataset_path}.")

    return cleaned_data, dataset_name, original_data, test_set

if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset"]["path"]
    separator = config["dataset"]["separator"]
    target_column = config["dataset"]["target_column"]

    cleaned_data, dataset_name, original_data, test_set = run_preprocessing(
        dataset_path, separator, target_column
    )

    print("Cleaned dataset shape:", cleaned_data.shape)
    print("Test set shape:", test_set.shape)
