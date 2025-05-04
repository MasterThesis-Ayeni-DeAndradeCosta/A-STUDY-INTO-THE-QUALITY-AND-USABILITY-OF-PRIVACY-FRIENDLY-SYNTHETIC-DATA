from math import log  # noqa: F401 – kept for consistency with existing imports
import os
import copy
import pandas as pd

from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing
from run_synthetic import run_synthetic
from output_utils.config_utils import generate_anonym_tag

def run_hybrid(
    train_raw_path: str,
    dataset_name: str,
    target_column: str,
    output_dir: str,
    config: dict,
    config_path: str,
    encoder=None,
    logger=None,
):
    """Create **hybrid synthetic datasets**.

    Steps
    -----
    1. Look for an *encoded anonymized* file produced by the standalone
       anonymization pipeline.
    2. If absent, look for the raw anonymized CSV and encode it.
    3. If *that* is also missing, run the Java/ARX anonymizer now and then encode.
    4. Call `run_synthetic()` on the encoded anonymized data to generate
       synthetic data for all enabled synthesizers.
    5. Save each hybrid dataset to the hybrid directory.
    """

    if logger:
        logger.info(" [HYBRID] Executing run_hybrid...")

    sep = config["dataset"]["separator"]
    anon_tag = generate_anonym_tag(config)

    # ---------- paths ----------
    anon_dir = os.path.abspath("datasets/anonymized")
    hyb_dir = os.path.abspath("datasets/hybrid")
    os.makedirs(anon_dir, exist_ok=True)
    os.makedirs(hyb_dir, exist_ok=True)

    base_name = f"{dataset_name}_{anon_tag}_anonymized"
    anon_csv = os.path.join(anon_dir, f"{base_name}.csv")
    enc_csv = os.path.join(anon_dir, f"{base_name}_encoded.csv")

    # ---------- obtain *encoded* anonymized data ----------
    if logger:
        logger.info(f" [HYBRID] Looking for encoded anonymized data in {enc_csv}")
    if os.path.exists(enc_csv):
        if logger:
            logger.info(f" [HYBRID ] Loaded cached encoded anonymous data from {enc_csv}")
        encoded_df = pd.read_csv(enc_csv, sep=sep)
    else:
        if not os.path.exists(anon_csv):
            if logger:
                logger.info(f" [HYBRID] No anonymized CSV - running ARX anonymization on {train_raw_path} …")
            success = run_anonymization(
                train_raw_path,
                config_path,
                anonymized_output_path=anon_csv,
            )
            if not success:
                if logger:
                    logger.error(" [HYBRID] Hybrid pipeline: anonymization failed.")
                raise RuntimeError("Hybrid pipeline: anonymization failed.")
            if logger:
                logger.info(f" [HYBRID] Anonymization of {train_raw_path} successful, data saved to {anon_csv}")

        if logger:
            logger.info(f" [HYBRID] Attempting post-processing of {anon_csv} …")
        enc_df, _, _, _ = run_postprocessing(
            anon_csv, sep, target_column,
            train_raw_path=train_raw_path,
            encoder=encoder,
            logger=logger
        )
        if enc_df is None:
            if logger:
                logger.error(" [HYBRID] Hybrid pipeline: post-processing failed.")
            raise RuntimeError("Hybrid pipeline: post-processing failed.")
        enc_df.to_csv(enc_csv, index=False, sep=sep)
        if logger:
            logger.info(f" [HYBRID] Post-processing successful!")
            logger.info(f" [HYBRID] Encoded anon data saved to {enc_csv}")
        encoded_df = enc_df

    if logger: 
        logger.info(f" [HYBRID] Starting synthetic generation for hybrid pipeline...")

    # ---------- generate hybrid datasets with all enabled synthesizers ----------
    hybrid_sets: dict[str, pd.DataFrame] = {}
    try:
        synth_datasets = run_synthetic(
            encoded_df,
            f"{dataset_name}_{anon_tag}",
            target_column,
            output_dir,
            config,
            logger=logger,
        )
    except Exception as e:
        if logger:
            logger.error(f"[HYBRID] run_synthetic failed: {e}")
        return {}

    for synth_name, df in synth_datasets.items():
        if df is None:
            if logger:
                logger.error(f" [HYBRID] {synth_name} returned no data.")
            continue
        
        if logger:
            logger.info(f"[HYBRID] Generated hybrid dataset shape for {synth_name}: {df.shape}")
            target_col = config["dataset"]["target_column"]
            if target_col in df.columns:
                class_dist = df[target_col].value_counts().to_dict()
                logger.info(f"[HYBRID] Target distribution for {synth_name}: {class_dist}")
            else:
                logger.warning(f"[HYBRID] Target column '{target_col}' missing from {synth_name} hybrid dataset!")

            if len(df) < 20:
                logger.warning(f"[HYBRID] {synth_name} hybrid dataset is very small: only {len(df)} rows.")
        
        out_csv = os.path.join(hyb_dir, f"{dataset_name}_{anon_tag}_{synth_name}_HYBRID.csv")
        df.to_csv(out_csv, index=False, sep=sep)
        if logger:
            logger.info(f" [HYBRID] Saved hybrid dataset to {out_csv}")
        hybrid_sets[synth_name] = df

    if logger:
        logger.info(f" [HYBRID] Hybrid pipeline complete — {len(hybrid_sets)} dataset(s) ready.")
        logger.info(f"[HYBRID] Total rows across all hybrid datasets: {sum(len(df) for df in hybrid_sets.values())}")

    return hybrid_sets