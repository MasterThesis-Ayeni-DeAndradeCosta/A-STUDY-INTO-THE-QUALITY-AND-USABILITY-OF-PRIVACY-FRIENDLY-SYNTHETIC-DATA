import os
import sys
import pandas as pd
from sdv.metadata import SingleTableMetadata
#from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from custom.synthesizer.base_synthesizer import BaseSynthesizer
from output_utils.config_utils import generate_anonym_tag
import importlib
import time  # NEW

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Define consistent path for storing synthesizers
SYNTHESIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",  "..", "artifacts", "synthesizers"))
os.makedirs(SYNTHESIZER_DIR, exist_ok=True)  # Ensure the directory exists

# Define the synthetic data storage directory
SYNTHETIC_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "synthetic"))
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)  # Ensure the directory exists

FILENAME_PARAM_MAP = {
    "CTGAN": ["epochs"],
    "TVAE": ["epochs"],
    "CustomSynthesizer": ["noise_factor"],
    # Add more synthesizers here as needed
}


def load_or_train_synthesizers(preprocessed_data, dataset_name, config, logger=None):
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

            # ✅ Safe param copy
            original_params = synth_info.get("params", {})
            params = original_params.copy()

            if logger:  # NEW
                logger.info(f"[SYNTHETIC] Preparing synthesizer: {synth_name}")  
                logger.info(f"[SYNTHETIC] Params from config: {params}")  

            module = importlib.import_module(module_name)
            synthesizer_class = getattr(module, class_name)

            # Ensure synthesizer follows BaseSynthesizer structure (for custom ones)
            if not issubclass(synthesizer_class, BaseSynthesizer) and "sdv" not in module_name:
                if logger:
                    logger.error(f" {class_name} does not implement BaseSynthesizer!")
                raise TypeError(f"❌ {class_name} does not implement BaseSynthesizer!")

            # Define synthesizer file path
            #synthesizer_filename = f"{dataset_name}_{synth_name}_synthesizer.pkl"
            #params = synth_info.get("params", {})  #accidentally got parms twice
            synthesizer_filename = generate_synthesizer_filename(dataset_name, synth_name, params)

            synthesizer_path = os.path.join(SYNTHESIZER_DIR, synthesizer_filename)

            #this code most likely caused the issue, sdv always defaulted to 300 epochs
            # Extract number of epochs if requireds
            # if "epochs" in params and class_name in ["CTGANSynthesizer", "TVAESynthesizer"]:
            #     epochs = params["epochs"]
            #     del params["epochs"]  # Remove epochs from params to avoid passing it to GaussianCopula
            # else:
            #     epochs = None  # Not needed for GaussianCopula
            
            # ✅ Safely extract epochs if it's in the config and supported by the class
            epochs = None
            if class_name in ["CTGANSynthesizer", "TVAESynthesizer"]:
                epochs = params.get("epochs")
                if epochs is not None:
                    del params["epochs"]  # Remove it from params to avoid double passing

            

            # Load or train synthesizer
            if os.path.exists(synthesizer_path):
                print(f"✅ Found existing {synth_name} synthesizer: {synthesizer_path}")
                if logger:
                    logger.info(f"[SYNTHETIC] Found existing {synth_name} synthesizer: {synthesizer_path}")
                if hasattr(synthesizer_class, "load"):
                    synthesizer = synthesizer_class.load(synthesizer_path)
                    if hasattr(synthesizer, "metadata"):
                        synthesizer.metadata = metadata
                        metadata = synthesizer.metadata 
                else: 
                    print(f"⚠️ Warning: {synth_name} does not support loading. Training a new one...")
                    if logger:
                        logger.warning(f"[SYNTHETIC] {synth_name} does not support loading. Training a new one...")
                    synthesizer = synthesizer_class(metadata, **params)

            else:
                print(f"Training new {synth_name} synthesizer...")
                if logger:
                    logger.info(" [SYNTHETIC] No pre-trained synthesizer found")
                    logger.info(f"[SYNTHETIC] Training new {synth_name} synthesizer from scratch...")

                # Check if the synthesizer requires `epochs`
                init_params = synthesizer_class.__init__.__code__.co_varnames  # Get constructor args

                if epochs is not None and "epochs" in init_params:
                    # Synthesizer supports epochs and we want to set it
                    synthesizer = synthesizer_class(metadata, epochs=epochs, **params)
                    if logger:
                        logger.info(f"[SYNTHETIC] {synth_name} initialized with epochs: {epochs}")
                        logger.info(f"[SYNTHETIC] Training parameters for {synth_name}: {params}")
                else:
                    if epochs is not None:
                        print(f"⚠️ Warning: {synth_name} does not accept 'epochs', ignoring it.")
                        if logger:
                            logger.warning(f"[SYNTHETIC] {synth_name} does not accept 'epochs', ignoring it.")
                    synthesizer = synthesizer_class(metadata, **params)


                if logger :
                    logger.info(f"[SYNTHETIC] fitting {synth_name} ") 
                start_time = time.time()     
                synthesizer.fit(preprocessed_data)
                end_time = time.time() 
                if logger:
                    logger.info(f"[SYNTHETIC] Fitting complete for {synth_name}. Duration: {end_time - start_time:.2f} seconds.")  # NEW
                # ✅ Confirm how many epochs were actually used
                actual_epochs = getattr(synthesizer, "epochs", None)
                if actual_epochs is not None:
                    print(f"📢 [CONFIRM] {synth_name} actually trained with epochs = {actual_epochs}")
                    if logger:
                        logger.info(f"[SYNTHETIC] CONFIRMATION {synth_name} actually trained with epochs = {actual_epochs}")
                else:
                    print(f"⚠️ Could not determine actual epochs used for {synth_name}")
                    if logger:
                        logger.warning(f"[SYNTHETIC] WARNING Could not determine actual epochs used for {synth_name}")
                
                synthesizer.save(synthesizer_path)
                print(f"✅ {synth_name} trained and saved.")
                if logger:
                    logger.info(f"[SYNTHETIC] {synth_name} trained and saved.")

            trained_synthesizers[synth_name] = (synthesizer, metadata)

        except (ImportError, AttributeError, TypeError) as e:
            print(f"❌ Error loading synthesizer '{synth_name}': {e}")
            if logger:
                logger.error(f"[SYNTHETIC] Error loading synthesizer '{synth_name}': {e}")

    if not trained_synthesizers:
        if logger:
            logger.error("No valid synthesizer enabled in config.")
        raise ValueError("No valid synthesizer enabled in config.")

    return trained_synthesizers, metadata


def generate_synthetic_datasets(preprocessed_data, dataset_name, config, logger=None):
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
    if logger:
        logger.info(f"[SYNTHETIC] Processing dataset: {dataset_name}")
        logger.info(f"[SYNTHETIC] Original dataset size: {len(preprocessed_data)} rows")

    # Load or train multiple synthesizers
    trained_synthesizers, metadata = load_or_train_synthesizers(preprocessed_data, dataset_name, config, logger=logger)

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
        elif num_generated_rows == "multiple":
            multiplier = synth_config.get("row_multiplier", 1.0)
            num_rows_to_generate = int(len(preprocessed_data) * multiplier)
        else:
            print(f"⚠️ Warning: Invalid num_generated_rows value '{num_generated_rows}' for {synth_name}. Defaulting to original dataset size.")
            if logger:
                logger.warning(f"[SYNTHETIC] Invalid num_generated_rows value '{num_generated_rows}' for {synth_name}. Defaulting to original dataset size.")
            num_rows_to_generate = len(preprocessed_data)

        print(f"\nUsing {synth_name} synthesizer to generate {num_rows_to_generate} synthetic rows.")
        # Log synth start
        if logger:
            logger.info(f"[SYNTHETIC] Using {synth_name} to generate {num_rows_to_generate} rows...")
        synthetic_data = synthesizer.sample(num_rows=num_rows_to_generate)
        print(f"✅ Synthetic data generated successfully: {len(synthetic_data)} rows created.")
        if logger:
            logger.info(f"[SYNTHETIC] {synth_name} completed: {len(synthetic_data)} rows generated.")

        
        target_col = config["dataset"]["target_column"]  # NEW
        if target_col in synthetic_data.columns:  # NEW
            counts = synthetic_data[target_col].value_counts().to_dict()  # NEW
            if logger :
                logger.info(f"[SYNTHETIC] Target class distribution for {synth_name}: {counts}")  # NEW
        else:  # NEW
            if logger : 
                logger.warning(f"[SYNTHETIC] {synth_name} generated data without target column '{target_col}'!")  # NEW

        synthetic_datasets[synth_name] = synthetic_data
        
    return synthetic_datasets, metadata

def generate_synthesizer_filename(dataset_name, synth_name, params):
    """
    Generate a synthesizer filename by appending the synthesizer name and training params
    to the full dataset name (which may already include hybrid or anonym tags).

    Args:
        dataset_name (str): Full dataset name, including any anonym/hybrid tags
        synth_name (str): e.g. 'CTGAN'
        params (dict): Training parameters like {"epochs": 50}

    Returns:
        str: Filename like 'loan_CTGAN_epochs50_synthesizer.pkl' or
             'loan_k3_l2_sup05_CTGAN_epochs50_synthesizer.pkl'
    """
    suffix_parts = [synth_name]

    keys_to_include = FILENAME_PARAM_MAP.get(synth_name, [])
    for key in keys_to_include:
        if key in params:
            value = params[key]
            if isinstance(value, float):
                value = str(value).replace('.', 'p')
            suffix_parts.append(f"{key}{value}")

    suffix = "_".join(suffix_parts)
    return f"{dataset_name}_{suffix}_synthesizer.pkl"




