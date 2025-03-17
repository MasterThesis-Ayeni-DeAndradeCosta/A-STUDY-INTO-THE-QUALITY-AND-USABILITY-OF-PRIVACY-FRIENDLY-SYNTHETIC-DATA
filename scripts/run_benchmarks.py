import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add `src/` to Python path
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0,  src_path)

import yaml
import pandas as pd
# Import necessary modules
from run_preprocessing import run_preprocessing
from run_utility import run_utility  
from run_synthetic import run_synthetic 
from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing 


from synthetic_pipeline.data_synthesis import generate_synthetic_datasets
from synthetic_pipeline.synthetic_evaluation import evaluate_synthetic_data, compare_data_distributions
#from modelOperations.model_training import evaluate_models
from modelOperations.model_evaluation import evaluate_models
from visualization.result_visualization import visualize_model_performance

# Import logging, output management, and report generation utilities
from output_utils.output_manager import create_output_directory
from output_utils.logger import setup_logger
from output_utils.report_generator import (
    save_preprocessing_report, 
    save_model_performance, 
    save_synthetic_data_report,
    save_yaml_config,
    save_synthetic_data_evaluation
)
from output_utils.visualization_saver import save_model_performance_graph

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
    """Executes the full benchmarking pipeline, saving logs, reports, and visualizations."""

    # Load configuration
    config = load_config()
    # Dataset parameters
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["dataset"]["path"]))
    
    separator = config["dataset"]["separator"]
    target_column = config["dataset"]["target_column"]

    #flags
    enable_synthetic = config["synthesis"]["enable_synthetic_generation"]
    enable_utility = config["utility"].get("enable_utility_evaluation", False)

    # Create formatted output directory
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_dir = create_output_directory(dataset_name)
    save_yaml_config(output_dir, config)
    # Setup logging
    logger = setup_logger(output_dir)
    logger.info(f" Benchmarking started for dataset: {dataset_name}")

    # Step 1: Run Preprocessing and Get Cleaned Data**
    cleaned_data, dataset_name, original_data, test_set = run_preprocessing(dataset_path, separator, target_column)
    cleaned_dataset_path = os.path.abspath(f"datasets/cleaned/{dataset_name}_cleaned.csv")

    if original_data is not None:
        save_preprocessing_report(output_dir, dataset_name, original_data, cleaned_data,  test_set)
    print("\nPreprocessing completed.")

    # Step 1.5: Run Anonymization (Flag handled internally)
    success = run_anonymization(cleaned_dataset_path)

    # After successful anonymization
    if success:
        # Path to anonymized data
        anonymized_dataset_path = os.path.abspath(f"datasets/anonymized/{dataset_name}_anonymized.csv")
        # Step 1.75: Post-process anonymized dataset (encoding)
        postprocessed_data = run_postprocessing(anonymized_dataset_path, separator, target_column)
    else:
        print("❌ Anonymization failed.")
        logger.error("Anonymization failed.")
        return None
    
    # Step 2 : Synthetic Data Generation
    synthetic_datasets, metadata = run_synthetic(cleaned_data, dataset_name, target_column, output_dir, config)  #disabled flag handled in run_synthetic

    if enable_utility:
        # Step 3: Run Utility for ML Training
        trained_models, X_test_original, y_test_original, datasets = run_utility(
            cleaned_data, synthetic_datasets, test_set, target_column, enable_synthetic, config
        )
        # Step 4: Evaluate Models
        print("\n Evaluating models...")       
        results_df = evaluate_models(trained_models, X_test_original, y_test_original, datasets, config)
        save_model_performance(output_dir, results_df)
        logger.info("Model Training and Evaluation completed.")

        # Step 5: Visualize results
        visualize_model_performance(results_df, dataset_name, output_dir)
        logger.info(" Visualizations saved.")
        logger.info(" Benchmarking Completed. Results saved in output folder.")
        print("\n Benchmarking Completed.")
        return results_df
    else:
        print("\nUtility Evaluation Skipped (Disabled in Configuration).")
        logger.info("Utility Evaluation Skipped (disabled in configuration).")
        results_df = None  # Return None to indicate no results were generated
    logger.info("Benchmarking Completed. Results saved in output folder.")
    print("\nBenchmarking Completed.")

    return results_df



    # # Step 3: Run Utility for ML Training
    # trained_models, X_test_original, y_test_original, datasets = run_utility(
    # cleaned_data, synthetic_datasets, test_set, target_column, enable_synthetic, config
    # )

    # # Step 4: Evaluate Models
    # print("\n Evaluating models...")
    # results_df = evaluate_models(trained_models, X_test_original, y_test_original, datasets, config)
    # save_model_performance(output_dir, results_df)
    # logger.info("Model Training and Evaluation completed.")

    # # Step 5: Visualize results
    # visualize_model_performance(results_df, dataset_name, output_dir)
    # logger.info(" Visualizations saved.")
    # logger.info(" Benchmarking Completed. Results saved in output folder.")

    # print("\n Benchmarking Completed.")
    # return results_df

if __name__ == "__main__":
    run_benchmarks()
