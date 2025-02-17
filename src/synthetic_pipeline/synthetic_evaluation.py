from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot


def evaluate_synthetic_data(original_data, synthetic_data, metadata, target_column, dataset_name):
    """
    Evaluates synthetic data quality and generates a diagnostic report.

    Parameters:
    - original_data (DataFrame): Original dataset.
    - synthetic_data (DataFrame): Generated synthetic dataset.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    - target_column (str): Column to visualize in diagnostic plots.
    - dataset_name (str): Name of the dataset being evaluated.
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
