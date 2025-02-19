import pandas as pd
from modelOperations.ml_data_preprocessing import prepare_original_data, prepare_synthetic_data
from modelOperations.model_training import train_models

def run_utility(encoded_data, synthetic_data, target_column, test_size, enable_synthetic, config):
    """
    Handles dataset preparation and model training.

    Parameters:
    - encoded_data (DataFrame): The cleaned dataset.
    - synthetic_data (DataFrame or None): The generated synthetic dataset.
    - target_column (str): Target variable for ML models.
    - test_size (float): Proportion of dataset for testing.
    - enable_synthetic (bool): Whether to include synthetic data in training.
    - config (dict): Configuration dictionary.

    Returns:
    - trained_models (dict): Dictionary of trained models.
    - X_test_original (DataFrame): Test set from original data.
    - y_test_original (Series): Target values for test set.
    - datasets (dict): Dictionary containing training datasets.
    """

    print(" Preparing datasets for machine learning...")

    # Split original dataset into training and test sets
    X_train_original, X_test_original, y_train_original, y_test_original = prepare_original_data(
        encoded_data, target_column, test_size=test_size
    )

    # Prepare synthetic dataset
    X_synthetic, y_synthetic = None, None
    if enable_synthetic and synthetic_data is not None:
        X_synthetic, y_synthetic = prepare_synthetic_data(synthetic_data, target_column)

    print(f" Original Data - Training Size: {len(X_train_original)}, Testing Size: {len(X_test_original)}")
    if X_synthetic is not None:
        print(f"ynthetic Data - Training Size: {len(X_synthetic)}")

    print("\nTraining models...")

    datasets = {"Original": (X_train_original, y_train_original)}
    if enable_synthetic and X_synthetic is not None:
        datasets["Synthetic"] = (X_synthetic, y_synthetic)

    trained_models = train_models(datasets, config)

    print("\n Model Training Completed.")
    return trained_models, X_test_original, y_test_original, datasets  # Pass back for evaluation

if __name__ == "__main__":
    print(" This script is not meant to be run directly. It should be imported and used in `run_benchmarks.py`.")
