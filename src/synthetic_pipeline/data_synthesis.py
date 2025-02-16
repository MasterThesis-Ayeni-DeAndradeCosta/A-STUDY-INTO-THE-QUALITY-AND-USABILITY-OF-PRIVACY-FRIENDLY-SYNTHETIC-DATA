import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# Define consistent path for storing synthesizers
SYNTHESIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthesizers"))
os.makedirs(SYNTHESIZER_DIR, exist_ok=True)  # Ensure the directory exists

# Define the synthetic data storage directory
SYNTHETIC_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "synthetic"))
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)  # Ensure the directory exists

def load_or_train_synthesizer(preprocessed_data, dataset_name):
    """
    Loads an existing trained synthesizer or trains a new one.

    Parameters:
    - preprocessed_data (DataFrame): The dataset after preprocessing.
    - dataset_name (str): Name of the dataset.

    Returns:
    - synthesizer (CTGANSynthesizer): Trained model.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    """
    # Initialize metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(preprocessed_data)

    # Define synthesizer path
    synthesizer_path = os.path.join(SYNTHESIZER_DIR, f"{dataset_name}_synthesizer.pkl")

    # Check if synthesizer trained with this data already exists, if so, use it, if not train the synthesizer with that data
    if os.path.exists(synthesizer_path):
        print(f"\n Found existing synthesizer for {dataset_name} at: {synthesizer_path}")
        print("Loading synthesizer...")
        synthesizer = CTGANSynthesizer.load(synthesizer_path)
        print("Synthesizer loaded successfully")
    else:
        print(f"\n No existing synthesizer for {dataset_name} found.")
        print("Training new synthesizer... (this may take a while)")
        synthesizer = CTGANSynthesizer(metadata)
        synthesizer.fit(preprocessed_data)
        synthesizer.save(synthesizer_path)
        print(" New synthesizer trained and saved successfully.")

    return synthesizer, metadata

def generate_synthetic_data(preprocessed_data, dataset_name, test_size=0.2):
    """
    Generates synthetic data for a given dataset.
    
    Parameters:
    - preprocessed_data (DataFrame): The dataset after preprocessing.
    - dataset_name (str): Name of the dataset (extracted beforehand).
    - test_size (float): Proportion of data to exclude from synthetic generation.
    
    Returns:
    - synthetic_data (DataFrame): A Pandas DataFrame containing the synthetic data.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    """
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Original dataset size: {len(preprocessed_data)} rows")

    # Load or train the synthesizer
    synthesizer, metadata = load_or_train_synthesizer(preprocessed_data, dataset_name)

    # Generate synthetic data
    print(f"\nUsing synthesizer '{dataset_name}_synthesizer' to generate synthetic data for dataset '{dataset_name}'")
    synthetic_data = synthesizer.sample(num_rows=int(len(preprocessed_data) * (1 - test_size)))
    print(f"Synthetic data generated successfully: {len(synthetic_data)} rows created")

    # Save synthetic data to datasets/synthetic
    synthetic_data_path = os.path.join(SYNTHETIC_DATA_DIR, f"{dataset_name}_synthetic.csv")
    synthetic_data.to_csv(synthetic_data_path, index=False)
    print(f"Synthetic data saved to {synthetic_data_path}")

    return synthetic_data, metadata
