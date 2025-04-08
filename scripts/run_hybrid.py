import os
import pandas as pd

from run_synthetic import run_synthetic
from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing

def run_hybrid(cleaned_data, dataset_name, target_column, output_dir, config):
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

    # Step 1: Run Synthetic
    synthetic_datasets, _ = run_synthetic(
        cleaned_data,
        dataset_name,
        target_column,
        output_dir,
        config
    )

    synthesizer_name = hybrid_cfg.get("synthesizer")
    if synthesizer_name is None:
        raise ValueError("Hybrid config must specify a 'synthesizer'.")

    synth_df = synthetic_datasets.get(synthesizer_name)
    if synth_df is None:
        raise RuntimeError(f"Synthesizer '{synthesizer_name}' did not return a dataset.")

    # Save synthetic CSV so it can be passed to run_anonymization
    synth_csv_path = f"datasets/synthetic/{dataset_name}_{synthesizer_name}_HYBRID.csv"
    os.makedirs(os.path.dirname(synth_csv_path), exist_ok=True)
    synth_df.to_csv(synth_csv_path, index=False)

    # Step 2: Run Anonymization on synthetic CSV
    success = run_anonymization(synth_csv_path, config)
    if not success:
        raise RuntimeError("Hybrid anonymization failed.")

    # Step 3: Post-process the anonymized result
    anonymized_path = f"datasets/anonymized/{dataset_name}_anonymized.csv"
    if not os.path.exists(anonymized_path):
        raise FileNotFoundError(f"Expected anonymized file not found: {anonymized_path}")

    separator = config["dataset"]["separator"]
    postprocessed_data, _, _ = run_postprocessing(anonymized_path, separator, target_column)

    return postprocessed_data
