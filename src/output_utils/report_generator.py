import os
from sympy import true
from tabulate import tabulate
import yaml
import pandas as pd

import os

def save_preprocessing_report(output_dir, dataset_name, original_data, processed_data, test_set,
                              handle_missing_strategy, test_size, encoding_type=None, encoding_map=None):
    """
    Saves preprocessing statistics in a structured, readable format.

    Parameters:
    - output_dir (str): Directory where the report will be saved.
    - dataset_name (str): Name of the dataset.
    - original_data (DataFrame): The raw dataset before preprocessing.
    - processed_data (DataFrame): The dataset after preprocessing.
    - test_set (DataFrame): The test set extracted from the data.
    - handle_missing_strategy (str): Strategy used for missing value handling.
    - test_size (float): Fraction used to split test set.
    - encoding_type (str, optional): Encoding type applied (e.g., "binary", "one-hot").
    - encoded_columns (list, optional): Categorical columns that were encoded.
    - encoding_map (dict, optional): Mapping from original categorical columns to new encoded columns.
    """
    report_path = os.path.join(output_dir, "preprocessing_report.txt")

    missing_before = original_data.isna().sum().sum()
    missing_after = processed_data.isna().sum().sum()
    handling_outcome = (
        "All missing values successfully imputed."
        if missing_after == 0 else f"{missing_after} missing values remain after preprocessing."
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Preprocessing Summary\n")
        f.write("---------------------\n")
        f.write(f"Dataset Name            : {dataset_name}\n")
        f.write(f"Missing Value Strategy  : {handle_missing_strategy}\n")
        f.write(f"Test Size (Fraction)    : {test_size:.2f}\n")

        if encoding_type:
            f.write(f"Encoding Type           : {encoding_type}\n")

        f.write("\nDataset Dimensions\n")
        f.write("------------------\n")
        f.write(f"Original Data           : {original_data.shape[0]} rows x {original_data.shape[1]} columns\n")
        f.write(f"Processed Data          : {processed_data.shape[0]} rows x {processed_data.shape[1]} columns\n")
        f.write(f"Test Set Size           : {test_set.shape[0]} rows\n")

        if handle_missing_strategy == "drop":
            rows_dropped = original_data.shape[0] - (processed_data.shape[0] + test_set.shape[0])
            f.write(f"Rows Dropped Due to Missing Values: {rows_dropped}\n")

        f.write("\nMissing Values\n")
        f.write("---------------\n")
        f.write(f"Total Missing (Before)  : {missing_before}\n")
        f.write(f"Total Missing (After)   : {missing_after}\n")
        f.write(f"Handling Outcome        : {handling_outcome}\n")

        f.write("\nMissing Values by Column (Before):\n")
        for col, count in original_data.isna().sum().items():
            if count > 0:
                f.write(f"  - {col}: {count}\n")

        if encoding_map:
            f.write("\nEncoding Mapping\n")
            f.write("----------------\n")
            f.write("Below is the mapping of original categorical columns to the new encoded binary columns:\n")
            for col, new_cols in encoding_map.items():
                joined = ', '.join(new_cols)
                f.write(f"- {col} ➝ {joined}\n")

    print(f"Preprocessing report saved at {report_path}")

def save_postprocessing_report(
    output_dir,
    dataset_name,
    encoding_type,
    encoding_map,
    anonymized_path=None,
    processed_path=None,
    encoded_path=None,
    logger=None
):
    """
    Saves a detailed report after postprocessing (normalization + encoding).

    Parameters:
    - output_dir (str): Directory where the report will be saved.
    - dataset_name (str): Name of the dataset.
    - encoding_type (str): Encoding method used (e.g., "binary").
    - encoding_map (dict): Mapping from original categorical columns to encoded columns.
    - anonymized_path (str): Path to the original ARX-anonymized CSV.
    - processed_path (str): Path to the normalized (cleaned) CSV.
    - encoded_path (str): Path to the final encoded CSV.
    - logger (Logger): Optional logger for logging save message.
    """
    report_path = os.path.join(output_dir, "postprocessing_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Postprocessing Report\n")
        f.write("======================\n\n")

        f.write("Dataset Information\n")
        f.write("-------------------\n")
        f.write(f"Dataset Name            : {dataset_name}\n")
        f.write(f"Encoding Type           : {encoding_type}\n")
        if anonymized_path:
            f.write(f"Anonymized Input Path   : {anonymized_path}\n")
        if processed_path:
            f.write(f"Processed Output Path   : {processed_path}\n")
        if encoded_path:
            f.write(f"Encoded Output Path     : {encoded_path}\n")

        f.write("\nNormalization Summary\n")
        f.write("---------------------\n")
        f.write("The following ARX artifacts were cleaned during normalization:\n")
        f.write("- Ranges like '[20-30]' → midpoint (e.g., 25.0)\n")
        f.write("- Top-coded values like '>=90' → numeric bound (e.g., 90.0)\n")
        f.write("- Suppressed values '*' → mode (categorical) or median (numeric)\n")
        f.write("- Unseen categorical values → replaced with mode from original training set\n")

        f.write("\nEncoding Mapping\n")
        f.write("----------------\n")
        if not encoding_map:
            f.write("⚠ No categorical columns were encoded.\n")
        else:
            f.write("Below is the mapping of original categorical columns to encoded binary columns:\n")
            for col, new_cols in encoding_map.items():
                joined = ', '.join(new_cols)
                f.write(f"- {col} -> {joined}\n")

    if logger:
        logger.info(f"Postprocessing report saved at: {report_path}")
    else:
        print(f"📄 Postprocessing report saved at {report_path}")


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
    return True


def save_anonymous_data_report(output_dir, dataset_name, anonymized_df, original_df=None):
    report_path = os.path.join(output_dir, "anonymous_data_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Anonymized Data Summary\n")
        f.write("=======================\n")
        f.write(f"Dataset Name      : {dataset_name}\n")
        f.write(f"Shape             : {anonymized_df.shape[0]} rows x {anonymized_df.shape[1]} columns\n")
        f.write(f"Total Missing     : {anonymized_df.isna().sum().sum()}\n\n")

        f.write("Column Overview\n")
        f.write("----------------\n")
        for col in anonymized_df.columns:
            n_unique = anonymized_df[col].nunique(dropna=False)
            n_missing = anonymized_df[col].isna().sum()
            f.write(f"- {col}: {n_unique} unique values, {n_missing} missing\n")

        f.write("\nSample Preview (first 5 rows)\n")
        f.write("--------------------------------\n")
        f.write(anonymized_df.head().to_string(index=False))
        f.write("\n\n")

        if original_df is not None:
            _compare_anonymized_to_original(f, anonymized_df, original_df)

    print(f"📄 Anonymous data report saved at {report_path}")

def _compare_anonymized_to_original(f, anonymized_df, original_df):
    f.write("Comparison with Original Data\n")
    f.write("==============================\n")

    orig_cols = set(original_df.columns)
    anon_cols = set(anonymized_df.columns)

    removed_cols = orig_cols - anon_cols
    added_cols = anon_cols - orig_cols

    f.write(f"Removed Columns: {', '.join(removed_cols) if removed_cols else 'None'}\n")
    f.write(f"New Columns Introduced: {', '.join(added_cols) if added_cols else 'None'}\n\n")

    # Check categorical differences
    cat_cols = original_df.select_dtypes(include='object').columns.intersection(anonymized_df.columns)

    for col in cat_cols:
        orig_vals = set(original_df[col].dropna().unique())
        anon_vals = set(anonymized_df[col].dropna().unique())
        new_vals = anon_vals - orig_vals

        if new_vals:
            f.write(f"⚠ Column '{col}' contains new values after anonymization: {new_vals}\n")
    f.write("\n")


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
