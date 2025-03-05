import os
from tabulate import tabulate
import yaml
import pandas as pd

def save_preprocessing_report(output_dir, dataset_name, original_data, processed_data,  test_set):
    """
    Saves preprocessing statistics.

    Parameters:
    - output_dir (str): Directory where the report will be saved.
    - dataset_name (str): Name of the dataset.
    - original_data (DataFrame): The raw dataset before preprocessing.
    - processed_data (DataFrame): The dataset after preprocessing.
    """
    report_path = os.path.join(output_dir, "preprocessing_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Original Rows: {original_data.shape[0]}, Columns: {original_data.shape[1]}\n")
        f.write(f"Processed Rows: {processed_data.shape[0]}, Columns: {processed_data.shape[1]}\n")
        f.write(f"Test Set Rows Reserved: {test_set.shape[0]}\n") 
        f.write(f"Missing Values Dropped/Imputed: {original_data.isna().sum().sum()} -> {processed_data.isna().sum().sum()}\n")

    print(f"📄 Preprocessing report saved at {report_path}")

def save_model_performance(output_dir, results_df):
    """
    Saves model evaluation results.

    Parameters:
    - output_dir (str): Directory where report will be saved.
    - results_df (DataFrame): DataFrame containing model evaluation metrics.
    """
    report_path = os.path.join(output_dir, "model_performance.txt")

    # Convert DataFrame to a formatted string table
    table = tabulate(results_df, headers="keys", tablefmt="grid")  
    
    with open(report_path, "w") as f:
        f.write(table)

    print(f"📄 Model performance saved at {report_path}")

def save_synthetic_data_evaluation(output_dir, diagnostic, quality_report):
    """
    Saves synthetic data evaluation metrics.

    Parameters:
    - output_dir (str): Directory where report will be saved.
    - diagnostic (dict): Diagnostic results from synthetic data evaluation.
    - quality_report (dict): Quality report from synthetic evaluation.
    """
    report_path = os.path.join(output_dir, "synthetic_data_evaluation.txt")
    
    with open(report_path, "w") as f:
        f.write("Synthetic Data Diagnostic Results:\n")
        f.write(str(diagnostic) + "\n\n")
        f.write("Synthetic Data Quality Report:\n")
        f.write(str(quality_report) + "\n")

    print(f"📄 Synthetic data evaluation saved at {report_path}")

def save_synthetic_data_report(output_dir, synthetic_datasets, quality_reports):
    """
    Saves a report detailing the generated synthetic datasets along with quality metrics.

    Parameters:
    - output_dir (str): The directory where the report will be saved.
    - synthetic_datasets (dict): Dictionary mapping synthesizer names to generated synthetic datasets.
    - quality_reports (dict): Dictionary mapping synthesizer names to their quality evaluation results.
    """
    report_path = os.path.join(output_dir, "synthetic_data_report.txt")

    with open(report_path, "w") as f:
        f.write("Synthetic Data Generation Report\n")
        f.write("=" * 50 + "\n\n")

        for synth_name, synthetic_data in synthetic_datasets.items():
            f.write(f"Synthesizer: {synth_name}\n")
            f.write(f"Generated Rows: {len(synthetic_data)}\n")
            f.write(f"Columns: {synthetic_data.shape[1]}\n")
            f.write("Sample (First 5 Rows):\n")
            f.write(synthetic_data.head().to_string(index=False) + "\n\n")

           # Add Summary Quality Metrics
            if synth_name in quality_reports:
                f.write(f"Quality Metrics for {synth_name}:\n")

                quality_report = quality_reports[synth_name]

                try:
                    overall_score = quality_report.get_score()
                    f.write(f"- Overall Quality Score: {overall_score:.4f}\n")
                except Exception as e:
                    f.write(f"- Error retrieving overall quality score: {str(e)}\n")

                try:
                    column_shapes_score = quality_report.get_details("Column Shapes").get("score", None)
                    if column_shapes_score is not None:
                        f.write(f"- Column Shapes Score: {column_shapes_score:.4f}\n")
                except Exception as e:
                    f.write(f"- Error retrieving column shapes score: {str(e)}\n")

                try:
                    column_pair_trends_score = quality_report.get_details("Column Pair Trends").get("score", None)
                    if column_pair_trends_score is not None:
                        f.write(f"- Column Pair Trends Score: {column_pair_trends_score:.4f}\n")
                except Exception as e:
                    f.write(f"- Error retrieving column pair trends score: {str(e)}\n")
            else:
                f.write("Error: No quality report available.\n")

            f.write("\n" + "=" * 50 + "\n\n")

    print(f"📄 Synthetic data report saved at {report_path}")

def save_yaml_config(output_dir, config):
    """
    Saves the YAML configuration used for the benchmark run.

    Parameters:
    - output_dir (str): The directory where the config file will be saved.
    - config (dict): The loaded YAML configuration dictionary.
    """
    config_path = os.path.join(output_dir, "benchmark_config.yaml")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"📄 YAML configuration saved at {config_path}")
