from math import log
import os
import pandas as pd

from run_synthetic import run_synthetic
from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing
import copy

def run_hybrid(train_raw_path, dataset_name, target_column, output_dir, config, config_path, logger=None):
    """
    Runs the hybrid pipeline: synthetic → anonymization.

    Parameters:
        cleaned_data (DataFrame): Cleaned and encoded training data.
        dataset_name (str): Name of the dataset.
        target_column (str): Target column for ML tasks.
        output_dir (str): Where outputs/logs should go.
        config (dict): Full YAML config dict.

    Returns:
        DataFrame: Encoded anonymized synthetic dataset (for utility training).
    """

    hybrid_cfg = config.get("hybrid", {})
    execution_order = hybrid_cfg.get("execution_order", "synthetic_first")

    if execution_order != "synthetic_first":
        raise NotImplementedError("Only 'synthetic_first' is supported in this hybrid version.")
    
    if logger:
        logger.info("Executing run_hybrid.....")

    if logger:
        logger.info("Loading unencoded training data for hybrid pipeline...")

    separator = config["dataset"]["separator"]
    try:
        cleaned_data = pd.read_csv(train_raw_path, sep=separator)
    except Exception as e:
        if logger:
            logger.error(f"Error loading unencoded training data: {e}")
        raise RuntimeError("Unable to run hybrid pipeline due to missing unencoded training data.")

    
    #extract synthesizer name from hybrid config
    synthesizer_name = hybrid_cfg.get("synthesizer")
    if synthesizer_name is None:
        raise ValueError("Hybrid config must specify a 'synthesizer'.")
    
     # Extract synthesizer config from synthesis section
    synth_config_original = config["synthesis"]["synthesizers"].get(synthesizer_name)
    if not synth_config_original:
        if logger:
            logger.error(f"Synthesizer '{synthesizer_name}' not found in synthesis config.")
        raise ValueError(f"Synthesizer '{synthesizer_name}' is not defined in synthesis config.")
    
    # clone it to avoid modifying the original config, ensure enable is set to True
    synth_config = copy.deepcopy(synth_config_original)
    synth_config["enabled"] = True

    # inject into a minimal config for only one synhtesizer to run_synthesizer
    hybrid_config = copy.deepcopy(config)
    hybrid_config["synthesis"] = {
        "enable_synthetic_generation": True,
        "synthesizers": {
            synthesizer_name: synth_config
        }
    }

    # Step 1: Run Synthetic
    synthetic_datasets = run_synthetic(
        cleaned_data,
        dataset_name,
        target_column,
        output_dir,
        hybrid_config
    )

    if logger:
        logger.info("Synthetic datasets generated for hybrid pipeline.")


    synth_df = synthetic_datasets.get(synthesizer_name)
    if synth_df is None:
        raise RuntimeError(f"Synthesizer '{synthesizer_name}' did not return a dataset.")

    # Save synthetic CSV in hybrid folder
    hybrid_dir = os.path.abspath("datasets/hybrid")
    os.makedirs(hybrid_dir, exist_ok=True)

    synth_csv_path = os.path.join(hybrid_dir, f"{dataset_name}_{synthesizer_name}_HYBRID.csv")
    synth_df.to_csv(synth_csv_path, index=False, sep=separator)

    if logger:
        logger.info(f"Synthetic dataset saved to {synth_csv_path} for hybrid pipeline.")


    if logger:
        logger.info("Hybrid Pipeline : Running anonymization on synthetic dataset...")
    
    hybrid_tag = f"{dataset_name}_{synthesizer_name}_HYBRID"
    anonymized_output_path = f"datasets/anonymized/{hybrid_tag}_anonymized.csv"
    # Step 2: Run Anonymization on synthetic CSV
    success = run_anonymization(synth_csv_path, config_path,  anonymized_output_path=anonymized_output_path)
    if not success:
        print("❌ Hybrid anonymization failed.")
        if logger:
            logger.error("Hybrid anonymization failed.")
        raise RuntimeError("Hybrid anonymization failed.")

    # Step 3: Post-process the anonymized result
    anonymized_path = anonymized_output_path
    if logger:
        logger.info(f"Anonymized dataset path for hybrid pipeline: {anonymized_path}")
    if not os.path.exists(anonymized_path):
        if logger:
            logger.error(f"Expected anonymized file not found: {anonymized_path}")
        raise FileNotFoundError(f"Expected anonymized file not found: {anonymized_path}")

    
    if logger:
        logger.info("beginning postprocessing of anonymized data in hybrid pipeline...")
    postprocessed_data, _, _ = run_postprocessing(anonymized_path, separator, target_column)

    if logger:
        logger.info("Hybrid pipeline completed successfully.")

    return postprocessed_data
