import os
from tabulate import tabulate

def save_preprocessing_report(output_dir, dataset_name, original_data, processed_data):
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
