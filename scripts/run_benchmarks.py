from math import log
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
import argparse
import time

# Import necessary modules
from run_preprocessing import run_preprocessing
from run_utility import run_utility  
from run_synthetic import run_synthetic 
from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing 
from run_analysis import run_analysis
from run_hybrid import run_hybrid


from modelOperations.model_evaluation import evaluate_models
from visualization.result_visualization import visualize_model_performance
from output_utils.output_manager import create_output_directory
from output_utils.logger import setup_logger
from output_utils.report_generator import (
    save_preprocessing_report, 
    save_model_performance, 
    save_yaml_config,
    save_postprocessing_report,
    save_anonymous_data_report
    )
import warnings
import logging


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

def generate_anonym_tag(config):
    models = config["anonymization"]["models"]
    k = models.get("k_anonymity", "kX")
    l = models.get("l_diversity", {}).get("value", "lX")
    sup = config["anonymization"].get("suppression_limit", "supX")
    return f"k{k}_l{l}_sup{int(sup * 100):02d}"  # e.g., k3_l2_sup05

def capture_warnings_in_logger(logger):
    def warning_to_log(message, category, filename, lineno, file=None, line=None):
        logger.warning(f"[{category.__name__}] {message} (from {filename}:{lineno})")
    warnings.showwarning = warning_to_log

def capture_exceptions_in_logger(logger):
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let the keyboard interrupt go through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def run_benchmarks(config_path="configs/benchmark_config.yaml"):
    """Executes the full benchmarking pipeline, saving logs, reports, and visualizations."""
    # Load configuration
    config = load_config(config_path=config_path)
    # Dataset parameters
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", config["dataset"]["path"]))
    separator = config["dataset"]["separator"]
    target_column = config["dataset"]["target_column"]
    handle_missing_strategy = config["preprocessing"]["handle_missing_values"]
    test_size = config["dataset"]["test_size"]
    encoding_type = config["preprocessing"]["encoding_type"]
    #flags
    enable_synthetic = config["synthesis"]["enable_synthetic_generation"]
    enable_anonymization = config["anonymization"]["enable_anonymization"]
    enable_utility = config["utility"].get("enable_utility_evaluation", False)
    enable_hybrid = config["hybrid"].get("enable_hybrid", False)
    analysis_config = config.get("analysis", {})

    # Create formatted output directory
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    if "output_dir" in config:
        output_dir = config["output_dir"]
    else:
    # Use default singular directory if not provided
        output_dir = create_output_directory(dataset_name, base_dir="singular")

    save_yaml_config(output_dir, config)
    # Setup logging
    logger = setup_logger(output_dir)
    capture_warnings_in_logger(logger)
    capture_exceptions_in_logger(logger)

    logger.info(f"Loaded config file: {config_path}")
    logger.info(f" Benchmarking started for dataset: {dataset_name}")
    logger.info(f"Using dataset path: {dataset_path}")
    logger.info(f"Target column: {target_column}, separator: '{separator}'")

    logger.info(f"Configuration loaded: enable_anonymization = {enable_anonymization}, enable_synthesis = {enable_synthetic}, enable_hybrid = {enable_hybrid}, enable_utility = {enable_utility}, delete_after_evaluation = {config['anonymization'].get('delete_after_evaluation', False)}")
    logger.info("Running preprocessing...")
    # Step 1: Run Preprocessing and Get Cleaned Data**
    cleaned_data, dataset_name, original_data, test_set, train_raw_path, encoding_map, encoder = run_preprocessing(dataset_path, separator, target_column, config_path=config_path , logger=logger)
    #cleaned_dataset_path = os.path.abspath(f"datasets/cleaned/{dataset_name}_cleaned.csv")
    
    if original_data is not None:
        save_preprocessing_report(output_dir, dataset_name, original_data, cleaned_data,  test_set, handle_missing_strategy, test_size, encoding_type, encoding_map)
        logger.info(f"Preprocessing completed. Rows before: {len(original_data)}, after: {len(cleaned_data)}")
        logger.info(f"Preprocessed data saved to: {os.path.join(output_dir, "preprocessing_report.txt")}")

    print("\nPreprocessing completed.")

    logger.info(f"enable_anonymization = {enable_anonymization}")

    # Step 2: Anonymization
    postprocessed_data = None
    if enable_anonymization:
        anonym_tag = generate_anonym_tag(config)
        anonymized_output_path = os.path.abspath(f"datasets/anonymized/{dataset_name}_{anonym_tag}_anonymized.csv")
        logger.info(f"Starting Anonymization Pipeline with output path: {anonymized_output_path}.......")
        success = run_anonymization(train_raw_path, config_path=config_path, anonymized_output_path=anonymized_output_path, logger=logger)
        #success = run_anonymization(train_raw_path, config_path=config_path, logger=logger)
        if success:
            #anonymized_dataset_path = os.path.abspath(f"datasets/anonymized/{dataset_name}_anonymized.csv")
            logger.info(f"Anonymized data saved to: {anonymized_output_path}")
            if os.path.exists(anonymized_output_path):
                df_anonymized = pd.read_csv(anonymized_output_path, sep=separator)
                save_anonymous_data_report(output_dir, dataset_name, df_anonymized, original_df=original_data)
                logger.info(f"Anonymous data report saved at: {os.path.join(output_dir, 'anonymous_data_report.txt')}")
                postprocessed_data, processed_output_path, encoded_output_path, anon_encoding_map = run_postprocessing(anonymized_output_path, separator, target_column, train_raw_path=train_raw_path,encoder=encoder, logger=logger)

                if postprocessed_data is not None and anon_encoding_map is not None:
                    save_postprocessing_report(
                    output_dir,
                    dataset_name,
                    encoding_type,
                    anon_encoding_map,
                    anonymized_path=anonymized_output_path,
                    processed_path=processed_output_path,
                    encoded_path=encoded_output_path,
                    logger=logger
                    )
                    
            else:
                print(f"⚠️ Anonymized file not found: {anonymized_output_path}")
                logger.warning(f"Anonymized file not found: {anonymized_output_path}")
        else:
            print("❌ Anonymization failed.")
            logger.error("Anonymization Pipeline failed!")
    else:
        print("🔹 Anonymization Skipped (Disabled in Configuration).")
        logger.info("Anonymization Skipped (Disabled in Configuration).")
    

    logger.info(f"enable_synthetic_generation = {enable_synthetic}")

    # Step 3 : Synthetic Data Generation

    #synthetic_datasets = run_synthetic(cleaned_data, dataset_name, target_column, output_dir, config, logger=logger)  #disabled flag handled in run_synthetic
    #logger.info(f"Synthetic data generated with {len(synthetic_datasets)} synthesizers.")
    synthetic_datasets = {}
    if enable_synthetic:
        logger.info("Starting standard Synthetic Data Generation pipeline...")
        synthetic_datasets = run_synthetic(cleaned_data, dataset_name, target_column, output_dir, config, logger=logger)
        logger.info(f"Synthetic data generated with {len(synthetic_datasets)} synthesizer(s).")
    else:
        logger.info("Synthetic Data Generation Skipped (Disabled in Configuration).")

    
    # step 4: Run Hybrid if enabled
    hybrid_data = None
    if enable_hybrid:
        logger.info("Running Hybrid Pipeline...")
        try:
            start = time.time()
            hybrid_data = run_hybrid(
                train_raw_path,
                dataset_name,
                target_column,
                output_dir,
                config,
                config_path,
                logger=logger
            )
            duration = time.time() - start 
            logger.info("Hybrid pipeline completed successfully.")
            logger.info(f"Hybrid pipeline duration: {duration:.2f} seconds")
        except Exception as e:
            logger.error(f"Hybrid pipeline failed: {e}")
            print(f" Hybrid Pipeline failed: {e}")
    else:
        logger.info("Hybrid Pipeline skipped (disabled in configuration).")
        print("Hybrid pipeline skipped (disabled in configuration).")

    # Step 5: Run Utility for ML Training
    if enable_utility:
        logger.info("Starting Utility...")
        trained_models, X_test_original, y_test_original, datasets = run_utility(
            cleaned_data,
            test_set,
            target_column,
            enable_synthetic,
            config,
            synthetic_datasets=synthetic_datasets if enable_synthetic else None,
            anonymous_data=postprocessed_data if enable_anonymization else None,
            hybrid_data=hybrid_data if enable_hybrid else None,
            logger=logger
        )
 
        # Step 6: Evaluate Models
        print("\n Evaluating models...") 
        logger.info("Starting the Evaluating of models...") 
        eval_start = time.time()     
        results_df = evaluate_models(trained_models, X_test_original, y_test_original, datasets, config, logger=logger)
        eval_duration = time.time() - eval_start
        logger.info(f"Model evaluation completed in {eval_duration:.2f} seconds.")
        save_model_performance(output_dir, results_df)
        results_csv_path = os.path.join(output_dir, "model_performance.csv")
        results_df.to_csv(results_csv_path, index=False)
        evaluated_models = list(results_df['Model'].unique())
        logger.info(f"Evaluated models: {evaluated_models}")
        logger.info("Model Training and Evaluation completed.")

        # Step 7: Visualize results
        visualize_model_performance(results_df, dataset_name, output_dir)
        logger.info(" Visualizations saved.")
        logger.info(" Benchmarking Completed. Results saved in output folder.")
        print("\n Benchmarking Completed.")

        run_analysis(output_dir, analysis_config)

        if config["anonymization"].get("delete_after_evaluation", False):
            if enable_anonymization:
                anon_dir = os.path.abspath("datasets/anonymized")
                if os.path.exists(anon_dir):
                    try:
                        for file in os.listdir(anon_dir):
                            if file.startswith(dataset_name) and "_anonymized" in file and file.endswith(".csv"):
                                file_path = os.path.join(anon_dir, file)
                                os.remove(file_path)
                                print(f" Deleted anonymized file: {file_path}")
                                logger.info(f"Deleted anonymized file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete anonymized files: {e}")

        # Delete hybrid datasets if configured
        if config["hybrid"].get("delete_hybrid_after_evaluation", False):
            if enable_hybrid:
                hybrid_dir = os.path.abspath("datasets/hybrid")
                if os.path.exists(hybrid_dir):
                    try:
                        for file in os.listdir(hybrid_dir):
                            if file.startswith(dataset_name) and "_HYBRID.csv" in file:
                                file_path = os.path.join(hybrid_dir, file)
                                os.remove(file_path)
                                print(f"🗑️ Deleted hybrid dataset: {file_path}")
                                logger.info(f"Deleted hybrid dataset: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete hybrid files: {e}")


        return results_df
    else:
        print("\nUtility Evaluation Skipped (Disabled in Configuration).")
        logger.info("Utility Evaluation Skipped (disabled in configuration).")
        results_df = None  # Return None to indicate no results were generated
    logger.info("Benchmarking Completed. Results saved in output folder.")
    print("\nBenchmarking Completed.")

    if config["anonymization"].get("delete_after_evaluation", False):
        if enable_anonymization:
                anon_dir = os.path.abspath("datasets/anonymized")
                if os.path.exists(anon_dir):
                    try:
                        for file in os.listdir(anon_dir):
                            if file.startswith(dataset_name) and "_anonymized" in file and file.endswith(".csv"):
                                file_path = os.path.join(anon_dir, file)
                                os.remove(file_path)
                                print(f" Deleted anonymized file: {file_path}")
                                logger.info(f"Deleted anonymized file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete anonymized files: {e}")

        # Delete hybrid datasets if configured
    if config["hybrid"].get("delete_hybrid_after_evaluation", False):
        if enable_hybrid:
            hybrid_dir = os.path.abspath("datasets/hybrid")
            if os.path.exists(hybrid_dir):
                try:
                    for file in os.listdir(hybrid_dir):
                        if file.startswith(dataset_name) and "_HYBRID.csv" in file:
                                file_path = os.path.join(hybrid_dir, file)
                                os.remove(file_path)
                                print(f"🗑️ Deleted hybrid dataset: {file_path}")
                                logger.info(f"Deleted hybrid dataset: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete hybrid files: {e}")

    return results_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    run_benchmarks(config_path=args.config)
