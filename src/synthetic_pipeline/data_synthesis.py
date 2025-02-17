import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer

# Define consistent path for storing synthesizers
SYNTHESIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthesizers"))
os.makedirs(SYNTHESIZER_DIR, exist_ok=True)  # Ensure the directory exists

# Define the synthetic data storage directory
SYNTHETIC_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "synthetic"))
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)  # Ensure the directory exists

def load_or_train_synthesizer(preprocessed_data, dataset_name, config):
    """
    Loads an existing trained synthesizer or trains a new one.

    Parameters:
    - preprocessed_data (DataFrame): The dataset after preprocessing.
    - dataset_name (str): Name of the dataset.

    Returns:
    - synthesizer (CTGANSynthesizer): Trained model.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    """
    # Extract synthesis parameters from config
    synthesizer_type = config["synthesis"].get("synthesizer_type", "CTGAN").upper()
    num_epochs = config["synthesis"].get("num_epochs", 50)
    num_generated_rows = config["synthesis"].get("num_generated_rows", "same_as_original")
    custom_generated_rows = config["synthesis"].get("custom_generated_rows", 10000)

    # Ensure the synthesizer type is valid; if not, default to CTGAN
    valid_synthesizers = {"CTGAN", "TVAE", "GAUSSIANCOPULA"}
    if synthesizer_type not in valid_synthesizers:
        print(f"\nInvalid synthesizer type '{synthesizer_type}'. Defaulting to CTGAN.")
        synthesizer_type = "CTGAN"


    # Initialize metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(preprocessed_data)

    # Define synthesizer file path with type included in the name
    synthesizer_filename = f"{dataset_name}_{synthesizer_type}_synthesizer.pkl"
    synthesizer_path = os.path.join(SYNTHESIZER_DIR, synthesizer_filename)

    # Check if synthesizer trained with this data already exists, if so, use it, if not train the synthesizer with that data
    if os.path.exists(synthesizer_path):
        print(f"\n Found existing synthesizer for {dataset_name} at: {synthesizer_path}") 

        if synthesizer_type == "CTGAN":
            print(f"Loading {synthesizer_type} synthesizer...")
            synthesizer = CTGANSynthesizer.load(synthesizer_path)
        elif synthesizer_type == "TVAE":
            print(f"Loading {synthesizer_type} synthesizer...")
            synthesizer = TVAESynthesizer.load(synthesizer_path)
        else:
            print(f"Loading {synthesizer_type} synthesizer...")
            synthesizer = GaussianCopulaSynthesizer.load(synthesizer_path)

        print(f"{synthesizer_type} Synthesizer loaded successfully.")
        
    else:
        print(f"\n No existing {synthesizer_type} synthesizer found for '{dataset_name}'. Training a new {synthesizer_type} synthesizer...")
        print("Training new synthesizer... (this may take a while)")

         # Choose synthesizer type
        if synthesizer_type == "CTGAN":
            synthesizer = CTGANSynthesizer(metadata, epochs=num_epochs)
        elif synthesizer_type == "TVAE":
            synthesizer = TVAESynthesizer(metadata, epochs=num_epochs)
        else:
            synthesizer = GaussianCopulaSynthesizer(metadata)  # No epochs needed

        synthesizer.fit(preprocessed_data)
        synthesizer.save(synthesizer_path)
        print(" New synthesizer trained and saved successfully.")

    return synthesizer, metadata

def generate_synthetic_data(preprocessed_data, dataset_name, config):
    """
    Generates synthetic data and saves it in the datasets/synthetic directory.
    
    Parameters:
    - preprocessed_data (DataFrame): The dataset after preprocessing.
    - dataset_name (str): Name of the dataset (extracted beforehand).
    - config (dict): Configuration dictionary containing synthesis parameters.
    
    Returns:
    - synthetic_data (DataFrame): A Pandas DataFrame containing the synthetic data.
    - metadata (SingleTableMetadata): Metadata object used for synthesis.
    """
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Original dataset size: {len(preprocessed_data)} rows")

    # Load or train the synthesizer
    synthesizer, metadata = load_or_train_synthesizer(preprocessed_data, dataset_name, config)

    # Extract synthesizer type from config
    synthesizer_type = config["synthesis"].get("synthesizer_type", "CTGAN").upper()

    # Determine how many rows to generate
    num_generated_rows = config["synthesis"].get("num_generated_rows", "same_as_original")
    custom_generated_rows = config["synthesis"].get("custom_generated_rows", 10000)

    if num_generated_rows == "same_as_original":
        num_rows_to_generate = len(preprocessed_data)
    elif num_generated_rows == "custom":
        num_rows_to_generate = custom_generated_rows
    else:
        print(f"Invalid value for num_generated_rows: '{num_generated_rows}'. Defaulting to original dataset size.")
        num_rows_to_generate = len(preprocessed_data)


    # Generate synthetic data
    print(f"\nUsing {synthesizer_type} synthesizer to generate {num_rows_to_generate} synthetic rows.")
    synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)
    print(f"Synthetic data generated successfully: {len(synthetic_data)} rows created.")

    # Save synthetic data to datasets/synthetic using the same naming structure as synthesizers
    synthetic_data_filename = f"{dataset_name}_{synthesizer_type}_synthetic.csv"
    synthetic_data_path = os.path.join(SYNTHETIC_DATA_DIR, synthetic_data_filename)

    synthetic_data.to_csv(synthetic_data_path, index=False)
    print(f"Synthetic data saved to {synthetic_data_path}")

    return synthetic_data, metadata
