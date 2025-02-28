from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot
import os
import pandas as pd

def evaluate_synthetic_data(original_data, synthetic_data, metadata, target_column, dataset_name):
    """
    Evaluates synthetic data quality and generates a diagnostic report.

    Parameters:
    - original_data (DataFrame): Original dataset.
    - synthetic_data (DataFrame): Generated synthetic dataset.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    - target_column (str): Column to visualize in diagnostic plots.
    - dataset_name (str): Name of the dataset being evaluated.

    Returns:
    - quality_report (dict): Dictionary containing synthetic data quality metrics.
    """
    print(f"\nRunning diagnostic comparison for {dataset_name}...")
    diagnostic = run_diagnostic(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    print("Diagnostic Results:")
    print(diagnostic)

    print(f"\nEvaluating quality metrics for {dataset_name}...")
    quality_report = evaluate_quality(
        original_data,
        synthetic_data,
        metadata
    )
    print("Quality Report:")
    print(quality_report)

    return quality_report

    # print("\nAnalyzing column distributions...")
    # column_shapes = quality_report.get_details('Column Shapes')
    # print(column_shapes)

    # print("\nGenerating sample column distribution plot...")
    # fig = get_column_plot(
    #     real_data=original_data,
    #     synthetic_data=synthetic_data,
    #     column_name=target_column,
    #     metadata=metadata
    # )
    # fig.show()

def compare_column_distributions(original_data, synthetic_data):
    for column in original_data.columns:
        if original_data[column].dtype in ['int64', 'float64']:  # Numerical Columns
            print(f"\n Column: {column}")
            print(f"Original Mean: {original_data[column].mean()}, Std: {original_data[column].std()}")
            print(f"Synthetic Mean: {synthetic_data[column].mean()}, Std: {synthetic_data[column].std()}")


def compare_data_distributions(output_dir, original_data, synthetic_datasets, target_column):
    """
    Appends data distribution comparisons to the synthetic_data_report.txt.

    Parameters:
    - output_dir (str): Path to save the report.
    - original_data (DataFrame): The original dataset after preprocessing.
    - synthetic_datasets (dict): Dictionary mapping synthesizer names to synthetic datasets.
    - target_column (str): The name of the target column to exclude from numerical comparisons.
    """

    report_path = os.path.join(output_dir, "synthetic_data_report.txt")

    with open(report_path, "a", encoding="utf-8") as f:  # "a" appends to the existing file
        f.write("\n🔍 Data Distribution Comparison\n")
        f.write("=" * 50 + "\n\n")

        # Exclude the target column from numerical analysis
        numeric_columns = [col for col in original_data.columns if col != target_column]

        # Original Data Summary
        f.write("📌 Original Data Summary:\n")
        f.write(str(original_data[numeric_columns].describe()) + "\n\n")

        # Synthetic Data Summaries
        for synth_name, synthetic_data in synthetic_datasets.items():
            f.write(f"📌 Synthetic Data ({synth_name}) Summary:\n")
            f.write(str(synthetic_data[numeric_columns].describe()) + "\n\n")

            # Compute Mean Differences
            mean_diff = synthetic_data[numeric_columns].mean() - original_data[numeric_columns].mean()
            f.write("⚠ Mean Differences (Synthetic - Original):\n")
            f.write(str(mean_diff) + "\n\n")

            # Compute Standard Deviation Differences
            std_diff = synthetic_data[numeric_columns].std() - original_data[numeric_columns].std()
            f.write("⚠ Standard Deviation Differences (Synthetic - Original):\n")
            f.write(str(std_diff) + "\n\n")

        f.write("=" * 50 + "\n")

    print(f"📄 Data distribution comparison appended to {report_path}")