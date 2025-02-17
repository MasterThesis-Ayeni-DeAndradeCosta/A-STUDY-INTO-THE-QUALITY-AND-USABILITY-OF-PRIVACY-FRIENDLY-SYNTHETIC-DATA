import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add `src/` to Python path
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

#print("Current sys.path:", sys.path)

# Now import modules from src
import yaml
import pandas as pd
from preprocessing.data_loader import load_dataset
from preprocessing.missing_value_handler import handle_missing_values
from preprocessing.encoding import encode_categorical_features
from synthetic_pipeline.data_synthesis import generate_synthetic_data
from modelOperations.ml_data_preprocessing import prepare_original_data, prepare_synthetic_data
from modelOperations.model_training import train_models, evaluate_models
from visualization.result_visualization import visualize_model_performance


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

    #synthetic parameters
    enable_synthetic = config["synthesis"]["enable_synthetic_generation"]
    test_size = config["synthesis"]["test_size"]

    # Load dataset
    original_data, dataset_name = load_dataset(dataset_path, separator)

    # Handle missing values
    cleaned_data = handle_missing_values(original_data, strategy=handle_missing)

    # Encode categorical features
    if encoding_type:
        encoded_data = encode_categorical_features(cleaned_data, target_column)
    else:
        encoded_data = cleaned_data

    print("\nPreprocessing completed.")

     # **Synthetic Data Generation**
    if enable_synthetic:
        synthetic_data, metadata = generate_synthetic_data(encoded_data, dataset_name, config)
        print("\nSynthetic Data Generation Completed.")
    else:
        print("\nSynthetic Data Generation Skipped (Disabled in Configuration).")

    # **Step 2: Prepare Datasets for ML Training**
    print("\n Preparing datasets for machine learning...")

    # Split original dataset into training and test sets
    X_train_original, X_test_original, y_train_original, y_test_original = prepare_original_data(
        encoded_data, target_column, test_size=test_size
    )

    # Prepare synthetic dataset (not split into test sets)
    X_synthetic, y_synthetic = None, None
    if enable_synthetic and synthetic_data is not None:
        X_synthetic, y_synthetic = prepare_synthetic_data(synthetic_data, target_column)

    print(f"Original Data - Training Size: {len(X_train_original)}, Testing Size: {len(X_test_original)}")
    if X_synthetic is not None:
        print(f"Synthetic Data - Training Size: {len(X_synthetic)}")

    print("\n Training models...")

    datasets = {
        "Original": (X_train_original, y_train_original)
    }
    if enable_synthetic and X_synthetic is not None:
        datasets["Synthetic"] = (X_synthetic, y_synthetic)

    trained_models = train_models(datasets)

    # **Step 4: Evaluate Models**
    print("\n Evaluating models...")
    results_df = evaluate_models(trained_models, X_test_original, y_test_original, datasets)

    # Step 5: Visualize results
    visualize_model_performance(results_df, dataset_name)

    print("\n Benchmarking Completed.")
    return results_df

if __name__ == "__main__":
    run_benchmarks()
