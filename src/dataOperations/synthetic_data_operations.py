import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot


def generate_synthetic_data(original_dataset_path, separator=",",test_size=0.2):
    """
    Generates synthetic data for a given dataset.
    
    Parameters:
    - original_dataset_path (str): Path to the original dataset file.
    - separator (str): The delimiter used in the dataset (e.g., ',' or '\t'). Default is ','.
    
    Returns:
    - synthetic_data (DataFrame): A Pandas DataFrame containing the synthetic data.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    - original_data (DataFrame): Original data as a Pandas DataFrame.
    - dataset_name (str): Name of the dataset extracted from the file path.
    """
    # Define consistent path for storing synthesizers
    synthesizer_dir = "../src/synthesizers"

    # Extract dataset name from path (gets 'studentPerformance' from "../datasets/original/studentPerformance.csv")
    dataset_name = os.path.splitext(os.path.basename(original_dataset_path))[0]
    
    # Read the original dataset
    original_data = pd.read_csv(original_dataset_path, sep=separator)
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Original dataset size: {len(original_data)} rows")

    # Count how many rows we had initially
    original_count = len(original_data)

    # Drop rows with missing values
    original_data.dropna(inplace=True)

    # Calculate how many rows were dropped
    dropped_rows = original_count - len(original_data)

    # Print a message to see how many were dropped
    print(f"Dropped {dropped_rows} rows due to missing values")
    
    # Initialize metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_data)
    
    # Define synthesizer path
    synthesizer_path = os.path.join(synthesizer_dir, f"{dataset_name}_synthesizer.pkl")
    
    # Check if synthesizer trained with this data already exists, if so, use it, if not train the synthesizer with that data
    if os.path.exists(synthesizer_path):
        print(f"\nFound existing synthesizer for {dataset_name} at: {synthesizer_path}")
        print("Loading synthesizer...")
        synthesizer = CTGANSynthesizer.load(synthesizer_path)
        print("Synthesizer loaded successfully")
    else:
        print(f"\nNo existing synthesizer for {dataset_name} found.")
        print("Training new synthesizer... (this may take a while)")
        synthesizer = CTGANSynthesizer(metadata)
        synthesizer.fit(original_data)
        os.makedirs(synthesizer_dir, exist_ok=True)  # Ensure the directory exists
        synthesizer.save(synthesizer_path)
        print("New synthesizer trained and saved successfully")
    
    # Generate synthetic data
    print(f"\nUsing synthesizer '{dataset_name}_synthesizer' to generate synthetic data for dataset '{dataset_name}'")
    synthetic_data = synthesizer.sample(num_rows=int(len(original_data) * (1 - test_size)))
    print(f"Synthetic data generated successfully: {len(synthetic_data)} rows created")
    
    return synthetic_data, metadata, original_data, dataset_name

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
    # Run diagnostic comparison
    print(f"\nRunning diagnostic comparison for {dataset_name}...")
    diagnostic = run_diagnostic(
        real_data=original_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    print("Diagnostic Results:")
    print(diagnostic)

    # Quality evaluation
    print(f"\nEvaluating quality metrics for {dataset_name}...")
    quality_report = evaluate_quality(
        original_data,
        synthetic_data,
        metadata
    )
    print("Quality Report:")
    print(quality_report)

    # Get details of column shapes
    print("\nAnalyzing column distributions...")
    column_shapes = quality_report.get_details('Column Shapes')
    print(column_shapes)

    # Generate column distribution plot
    print("\nGenerating sample column distribution plot...")
    fig = get_column_plot(
        real_data=original_data,
        synthetic_data=synthetic_data,
        column_name=target_column,
        metadata=metadata
    )
    fig.show()