from math import log
import os
import pandas as pd
import copy

from run_synthetic import run_synthetic
from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing
from output_utils.config_utils import generate_anonym_tag


def run_hybrid(train_raw_path, dataset_name, target_column, output_dir, config, config_path, logger=None):
    """
    Runs the hybrid pipeline: anonymization → postprocessing → synthesis.

    Saves all files in datasets/hybrid/.

    Returns:
        pd.DataFrame: Hybrid synthetic data generated from anonymized → postprocessed data.
    """
    if logger:
        logger.info("Executing run_hybrid...")

    separator = config["dataset"]["separator"]
    hybrid_cfg = config.get("hybrid", {})

    # Create hybrid directory if not exists
    hybrid_dir = os.path.abspath("datasets/hybrid")
    os.makedirs(hybrid_dir, exist_ok=True)

    anonym_tag = generate_anonym_tag(config)
    hybrid_dataset_name = f"{dataset_name}_{anonym_tag}"

    # STEP 1: Run anonymization
    anonym_csv_path = os.path.join(hybrid_dir, f"{hybrid_dataset_name}.csv")
    success = run_anonymization(train_raw_path, config_path, anonymized_output_path=anonym_csv_path)
    if not success:
        if logger:
            logger.error("Hybrid pipeline failed during anonymization step.")
        raise RuntimeError("Hybrid pipeline failed during anonymization step.")
    if logger:
        logger.info(f" Anonymized file created at {anonym_csv_path}")

    # STEP 2: Postprocess (encode anonymized data)
    postprocessed_data, _, _ = run_postprocessing(anonym_csv_path, separator, target_column, logger=logger)

    encoded_path = os.path.join(hybrid_dir, f"{hybrid_dataset_name}_encoded.csv")
    if postprocessed_data is None:
        if logger:
            logger.error("Postprocessing failed: No encoded data returned.")
        raise RuntimeError("Hybrid pipeline failed during postprocessing.")
    postprocessed_data.to_csv(encoded_path, index=False, sep=separator)
    if logger:
        logger.info(f" Encoded anonymized dataset saved to: {encoded_path}")

    # STEP 3: Run synthesis using the saved encoded file (to avoid metadata mismatch)
    loaded_encoded_data = pd.read_csv(encoded_path, sep=separator)

    synthesizer_name = hybrid_cfg.get("synthesizer")
    if synthesizer_name is None:
        if logger:
            logger.error("Hybrid config did not specify a 'synthesizer'.")
        raise ValueError("Hybrid config must specify a 'synthesizer'.")

    synth_config_original = config["synthesis"]["synthesizers"].get(synthesizer_name)
    if not synth_config_original:
        raise ValueError(f"Synthesizer '{synthesizer_name}' not found in synthesis config.")

    # Construct mini-config with just one enabled synthesizer
    synth_config = copy.deepcopy(synth_config_original)
    synth_config["enabled"] = True
    hybrid_config = copy.deepcopy(config)
    hybrid_config["synthesis"] = {
        "enable_synthetic_generation": True,
        "synthesizers": {
            synthesizer_name: synth_config
        }
    }

    hybrid_synthetic_datasets = run_synthetic(
    loaded_encoded_data,
    hybrid_dataset_name,
    target_column,
    output_dir,
    hybrid_config,
    logger=logger
)

    
    hybrid_df = hybrid_synthetic_datasets.get(synthesizer_name)
    if hybrid_df is None:
        if logger:
            logger.error(f"Synthesizer '{synthesizer_name}' did not return a dataset.")
        raise RuntimeError(f"Synthesizer '{synthesizer_name}' did not return a dataset.")
    

    if logger:
        logger.info(f" Hybrid dataset generated successfully. Type: {type(hybrid_df)}, Shape: {hybrid_df.shape}")

    # Save the final dataset
    hybrid_output_path = os.path.join(hybrid_dir, f"{hybrid_dataset_name}_{synthesizer_name}_HYBRID.csv")
    hybrid_df.to_csv(hybrid_output_path, index=False, sep=separator)
    if logger:
        logger.info(f" Final hybrid synthetic dataset saved to: {hybrid_output_path}")

    return hybrid_df
