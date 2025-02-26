import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
#from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from custom.synthesizer.base_synthesizer import BaseSynthesizer
import importlib

# Define consistent path for storing synthesizers
SYNTHESIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthesizers"))
os.makedirs(SYNTHESIZER_DIR, exist_ok=True)  # Ensure the directory exists

# Define the synthetic data storage directory
SYNTHETIC_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "synthetic"))
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)  # Ensure the directory exists

def load_or_train_synthesizers(preprocessed_data, dataset_name, config):
    """
    Loads or trains multiple synthesizers dynamically.

    Parameters:
    - preprocessed_data (DataFrame): The dataset after preprocessing.
    - dataset_name (str): Name of the dataset.
    - config (dict): Loaded YAML configuration.

    Returns:
    - trained_synthesizers (dict): Dictionary of synthesizer instances with metadata.
    """
    synthesizers_config = config["synthesis"]["synthesizers"]
    trained_synthesizers = {}

    # Initialize metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(preprocessed_data)

    for synth_name, synth_info in synthesizers_config.items():
        if not synth_info.get("enabled", False):
            continue  # Skip disabled synthesizers

        try:
            # Dynamically import synthesizer class
            module_name = synth_info["import_path"]
            class_name = synth_info["class_name"]
            params = synth_info.get("params", {})

            module = importlib.import_module(module_name)
            synthesizer_class = getattr(module, class_name)

            # Ensure synthesizer follows BaseSynthesizer structure (for custom ones)
            if not issubclass(synthesizer_class, BaseSynthesizer) and "sdv" not in module_name:
                raise TypeError(f"❌ {class_name} does not implement BaseSynthesizer!")

            # Define synthesizer file path
            synthesizer_filename = f"{dataset_name}_{synth_name}_synthesizer.pkl"
            synthesizer_path = os.path.join(SYNTHESIZER_DIR, synthesizer_filename)

            # Extract number of epochs if required
            if "epochs" in params and class_name in ["CTGANSynthesizer", "TVAESynthesizer"]:
                epochs = params["epochs"]
                del params["epochs"]  # Remove epochs from params to avoid passing it to GaussianCopula
            else:
                epochs = None  # Not needed for GaussianCopula

            # Load or train synthesizer
            if os.path.exists(synthesizer_path):
                print(f"✅ Found existing {synth_name} synthesizer: {synthesizer_path}")
                if hasattr(synthesizer_class, "load"):
                    synthesizer = synthesizer_class.load(synthesizer_path)
                else: 
                    print(f"⚠️ Warning: {synth_name} does not support loading. Training a new one...")
                    synthesizer = synthesizer_class(metadata, **params)

            else:
                print(f"Training new {synth_name} synthesizer...")

                # Check if the synthesizer requires `epochs`
                if "epochs" in params:
                    init_params = synthesizer_class.__init__.__code__.co_varnames  # Get constructor args
                    if "epochs" in init_params:
                        synthesizer = synthesizer_class(metadata, epochs=params["epochs"], **params)
                    else:
                        print(f"⚠️ Warning: {synth_name} does not accept 'epochs', ignoring it.")
                        synthesizer = synthesizer_class(metadata, **params)
                else:
                     synthesizer = synthesizer_class(metadata, **params)
                     
                synthesizer.fit(preprocessed_data)
                synthesizer.save(synthesizer_path)
                print(f"✅ {synth_name} trained and saved.")

            trained_synthesizers[synth_name] = (synthesizer, metadata)

        except (ImportError, AttributeError, TypeError) as e:
            print(f"❌ Error loading synthesizer '{synth_name}': {e}")

    if not trained_synthesizers:
        raise ValueError("No valid synthesizer enabled in config.")

    return trained_synthesizers, metadata


def generate_synthetic_datasets(preprocessed_data, dataset_name, config):
    """
    Generates synthetic data using multiple synthesizers.

    Parameters:
    - preprocessed_data (DataFrame): The dataset after preprocessing.
    - dataset_name (str): Name of the dataset (extracted beforehand).
    - config (dict): Configuration dictionary containing synthesis parameters.

    Returns:
    - synthetic_datasets (dict): Dictionary mapping synthesizer names to synthetic datasets.
    """
    print(f"\nProcessing dataset: {dataset_name}")
    print(f"Original dataset size: {len(preprocessed_data)} rows")

    # Load or train multiple synthesizers
    trained_synthesizers, metadata = load_or_train_synthesizers(preprocessed_data, dataset_name, config)

    synthetic_datasets = {}
    
    for synth_name, (synthesizer, _) in trained_synthesizers.items():
        # Fetch synthesizer configurations
        synth_config = config["synthesis"]["synthesizers"][synth_name]

        # Get number of rows to generate from config (not from params)
        num_generated_rows = synth_config.get("num_generated_rows", "same_as_original")
        custom_generated_rows = synth_config.get("custom_generated_rows", 10000)

        # Determine number of rows to generate
        if num_generated_rows == "same_as_original":
            num_rows_to_generate = len(preprocessed_data)
        elif num_generated_rows == "custom":
            num_rows_to_generate = custom_generated_rows
        else:
            print(f"⚠️ Warning: Invalid num_generated_rows value '{num_generated_rows}' for {synth_name}. Defaulting to original dataset size.")
            num_rows_to_generate = len(preprocessed_data)

        print(f"\nUsing {synth_name} synthesizer to generate {num_rows_to_generate} synthetic rows.")
        synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)
        print(f"✅ Synthetic data generated successfully: {len(synthetic_data)} rows created.")

        # Save synthetic data
        synthetic_data_filename = f"{dataset_name}_{synth_name}_synthetic.csv"
        synthetic_data_path = os.path.join(SYNTHETIC_DATA_DIR, synthetic_data_filename)  # Save to datasets/synthetic

        synthetic_data.to_csv(synthetic_data_path, index=False)
        print(f" Synthetic data saved to {synthetic_data_path}")

        synthetic_datasets[synth_name] = synthetic_data
        
    return synthetic_datasets, metadata
