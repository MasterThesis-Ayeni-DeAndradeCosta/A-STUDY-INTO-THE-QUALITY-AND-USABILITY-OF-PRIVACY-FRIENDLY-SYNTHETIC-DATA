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
    logger.info(f"Loaded config file: {config_path}")
    logger.info(f" Benchmarking started for dataset: {dataset_name}")

    logger.info(f"Configuration loaded: enable_anonymization = {enable_anonymization}, enable_synthesis = {enable_synthetic}")
    logger.info("Running preprocessing...")
    # Step 1: Run Preprocessing and Get Cleaned Data**
    cleaned_data, dataset_name, original_data, test_set, train_raw_path, encoding_map  = run_preprocessing(dataset_path, separator, target_column, config_path=config_path , logger=logger)
    #cleaned_dataset_path = os.path.abspath(f"datasets/cleaned/{dataset_name}_cleaned.csv")
    
    if original_data is not None:
        save_preprocessing_report(output_dir, dataset_name, original_data, cleaned_data,  test_set, handle_missing_strategy, test_size, encoding_type, encoding_map)
        logger.info(f"Preprocessing completed. Rows before: {len(original_data)}, after: {len(cleaned_data)}")

    print("\nPreprocessing completed.")
    logger.info(f"enable_anonymization = {config.get('enable_anonymization')}")
    # Step 2: Anonymization
    postprocessed_data = None
    if enable_anonymization:
        success = run_anonymization(train_raw_path, config_path=config_path, logger=logger)
        if success:
            anonymized_dataset_path = os.path.abspath(f"datasets/anonymized/{dataset_name}_anonymized.csv")
            logger.info(f"Anonymized data saved to: {anonymized_dataset_path}")
            if os.path.exists(anonymized_dataset_path):
                df_anonymized = pd.read_csv(anonymized_dataset_path, sep=separator)
                save_anonymous_data_report(output_dir, dataset_name, df_anonymized, original_df=original_data)
                postprocessed_data, _, anon_encoding_map = run_postprocessing(anonymized_dataset_path, separator, target_column,logger=logger)

                if postprocessed_data is not None and anon_encoding_map:
                    save_postprocessing_report(output_dir, dataset_name, encoding_type, anon_encoding_map)
                else:
                    print(f"⚠️ Anonymized file not found: {anonymized_dataset_path}")
                    logger.warning(f"Anonymized file not found: {anonymized_dataset_path}")
        else:
            print("❌ Anonymization failed.")
            logger.error("Anonymization failed.")
    else:
        print("🔹 Anonymization Skipped (Disabled in Configuration).")
        logger.info("Anonymization Skipped (Disabled in Configuration).")
    

    logger.info(f"enable_synthetic_generation = {enable_synthetic}")
    # Step 3 : Synthetic Data Generation
    synthetic_datasets, metadata = run_synthetic(cleaned_data, dataset_name, target_column, output_dir, config, logger=logger)  #disabled flag handled in run_synthetic
    logger.info(f"Synthetic data generated with {len(synthetic_datasets)} synthesizers.")

    
    # step 4: Run Hybrid if enabled
    hybrid_data = None
    if enable_hybrid:
        logger.info("Running hybrid pipeline...")
        try:
            hybrid_data = run_hybrid(
                train_raw_path,
                dataset_name,
                target_column,
                output_dir,
                config,
                config_path,
                logger=logger
            )
            logger.info("Hybrid pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Hybrid pipeline failed: {e}")
            print(f" Hybrid pipeline failed: {e}")
    else:
        logger.info("Hybrid pipeline skipped (disabled in configuration).")
        print("Hybrid pipeline skipped (disabled in configuration).")
    # Step 5: Run Utility for ML Training
    if enable_utility:
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
        logger.info("Evaluating models...")      
        results_df = evaluate_models(trained_models, X_test_original, y_test_original, datasets, config)
        save_model_performance(output_dir, results_df)
        results_csv_path = os.path.join(output_dir, "model_performance.csv")
        results_df.to_csv(results_csv_path, index=False)
        logger.info("Model Training and Evaluation completed.")

        # Step 7: Visualize results
        visualize_model_performance(results_df, dataset_name, output_dir)
        logger.info(" Visualizations saved.")
        logger.info(" Benchmarking Completed. Results saved in output folder.")
        print("\n Benchmarking Completed.")

        run_analysis(output_dir, analysis_config)

        return results_df
    else:
        print("\nUtility Evaluation Skipped (Disabled in Configuration).")
        logger.info("Utility Evaluation Skipped (disabled in configuration).")
        results_df = None  # Return None to indicate no results were generated
    logger.info("Benchmarking Completed. Results saved in output folder.")
    print("\nBenchmarking Completed.")

    return results_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    run_benchmarks(config_path=args.config)
