import os
import pandas as pd

def load_dataset(original_dataset_path, separator=","):
    """
    Loads a dataset from a file path and extracts the dataset name.

    Parameters:
    - original_dataset_path (str): Path to the original dataset file.
    - separator (str): The delimiter used in the dataset (e.g., ',' or '\t'). Default is ','.

    Returns:
    - original_data (DataFrame): A Pandas DataFrame containing the dataset.
    - dataset_name (str): Extracted dataset name from the file path.
    """
    # Extract dataset name from path (e.g., "studentPerformance" from "../datasets/original/studentPerformance.csv")
    dataset_name = os.path.splitext(os.path.basename(original_dataset_path))[0]

    # Read the original dataset
    print(f"📂 Loading dataset from: {original_dataset_path}...")
    original_data = pd.read_csv(original_dataset_path, sep=separator)
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Original dataset size: {len(original_data)} rows")

    return original_data, dataset_name
