import os
import pandas as pd
import yaml
from preprocessing.data_loader import load_dataset
from preprocessing.missing_value_handler import handle_missing_values
from preprocessing.encoding import encode_categorical_features_train_test
from sklearn.model_selection import train_test_split
import joblib

# Load configuration
def load_config(config_path="configs/benchmark_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_preprocessing(dataset_path, separator, target_column,  config_path="configs/benchmark_config.yaml", logger=None):
    """
    Preprocesses dataset: missing value handling, split, encoding, saving files.
    Returns:
    - cleaned_train (DataFrame): Encoded train split.
    - dataset_name (str): Dataset name.
    - original_data (DataFrame): Raw dataset before preprocessing.
    - cleaned_test (DataFrame): Encoded test split.
    - train_raw_path (str): Path to unencoded train CSV for anonymization.
    - encoding_map (dict): Mapping from original categorical columns to encoded columns
    """
    config = load_config(config_path)
    test_size = config["dataset"]["test_size"]
    handle_missing = config["preprocessing"]["handle_missing_values"]
    encoding_type = config["preprocessing"]["encoding_type"]

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # File paths
    train_raw_path = f"datasets/train/{dataset_name}_train_raw.csv"
    cleaned_train_path = f"datasets/cleaned/{dataset_name}_cleaned.csv"
    cleaned_test_path = f"datasets/test/{dataset_name}_test_set.csv"
    encoder_path = f"artifacts/{dataset_name}_binary_encoder.pkl"
    os.makedirs("artifacts", exist_ok=True)

    os.makedirs("datasets/train", exist_ok=True)
    os.makedirs("datasets/cleaned", exist_ok=True)
    os.makedirs("datasets/test", exist_ok=True)

    # Check if preprocessed files exist
    if os.path.exists(train_raw_path) and os.path.exists(cleaned_train_path) and os.path.exists(cleaned_test_path) and os.path.exists(encoder_path):
        print(f"✅ Preprocessed files found. Skipping preprocessing.")
        if logger:
            logger.info(" [PREPROCESSING] Preprocessed files found. Skipping preprocessing.")
        cleaned_train = pd.read_csv(cleaned_train_path, sep=separator)
        cleaned_test = pd.read_csv(cleaned_test_path, sep=separator)
        original_data, _ = load_dataset(dataset_path, separator)
        encoder = joblib.load(encoder_path) #will be used in postprocessing
        # Encoding map will not be available if skipped (you can return None or reload if needed)
        return cleaned_train, dataset_name, original_data, cleaned_test, train_raw_path, None, encoder

    if logger:
        logger.info("[PREPROCESSING] No preprocessed files found. Starting full preprocessing pipeline...")

    # Full preprocessing flow
    original_data, _ = load_dataset(dataset_path, separator)
    cleaned_full = handle_missing_values(original_data, strategy=handle_missing)
    if logger:
        logger.info(f"[PREPROCESSING] Missing value handling strategy: '{handle_missing}' applied.")

    # Split raw data
    train_raw_df, test_raw_df = train_test_split(
        cleaned_full,
        test_size=test_size,
        stratify=cleaned_full[target_column],
        random_state=42
    )
    if logger:
        logger.info(f"[PREPROCESSING] Split data into train ({len(train_raw_df)} rows) and test ({len(test_raw_df)} rows). Test size: {test_size}")

    train_raw_df.to_csv(train_raw_path, index=False, sep=separator)  # Save raw train for anonymization

    # Encode
    #cleaned_train, encoding_map = encode_categorical_features(train_raw_df.copy(), target_column)
    #cleaned_test, _ = encode_categorical_features(test_raw_df.copy(), target_column)
    cleaned_train, cleaned_test, encoder, encoding_map = encode_categorical_features_train_test(train_raw_df.copy(), target_column, test_data=test_raw_df.copy())

    joblib.dump(encoder, encoder_path)
    

    # Save encoded versions
    cleaned_train.to_csv(cleaned_train_path, index=False, sep=separator)  
    cleaned_test.to_csv(cleaned_test_path, index=False, sep=separator)    

    print(f"✅ Saved raw train set to: {train_raw_path}")
    print(f"💾 Saved encoder to: {encoder_path}")
    print(f"✅ Saved encoded train set to: {cleaned_train_path}")
    print(f"✅ Saved encoded test set to: {cleaned_test_path}")
    if logger:
        logger.info(f" [PREPROCESSING] Saved raw train set to: {train_raw_path}")
        logger.info(f" [PREPROCESSING] Saved encoder to: {encoder_path}")
        logger.info(f" [PREPROCESSING]Saved encoded train set to: {cleaned_train_path}")
        logger.info(f" [PREPROCESSING] Saved encoded test set to: {cleaned_test_path}")

    if logger:
        logger.info("[PREPROCESSING] run_preprocessing finished executing.")

    

    return cleaned_train, dataset_name, original_data, cleaned_test, train_raw_path, encoding_map, encoder

if __name__ == "__main__":
    config = load_config()
    dataset_path = config["dataset"]["path"]
    separator = config["dataset"]["separator"]
    target_column = config["dataset"]["target_column"]

    cleaned_train, dataset_name, original_data, cleaned_test, train_raw_path, encoding_map, encoder = run_preprocessing(
        dataset_path, separator, target_column
    )

    print("Encoded Train Shape:", cleaned_train.shape)
    print("Encoded Test Shape:", cleaned_test.shape)
