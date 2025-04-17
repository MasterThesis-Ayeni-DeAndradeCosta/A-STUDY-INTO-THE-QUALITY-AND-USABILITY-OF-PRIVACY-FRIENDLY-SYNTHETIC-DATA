from math import log  # noqa: F401 – kept for consistency with existing imports
import os
import copy
import pandas as pd

from run_anonymization import run_anonymization
from run_postprocessing import run_postprocessing
from run_synthetic import run_synthetic
from output_utils.config_utils import generate_anonym_tag

# -----------------------------------------------------------------------------
# Hybrid pipeline: reuse existing anonymized data (if present) → encode →
# generate synthetic data with *every* synthesizer that has `enabled: true`.
# -----------------------------------------------------------------------------

def run_hybrid(
    train_raw_path: str,
    dataset_name: str,
    target_column: str,
    output_dir: str,
    config: dict,
    config_path: str,
    logger=None,
):
    """Create **hybrid synthetic datasets**.

    Steps
    -----
    1. Look for an *encoded anonymized* file produced by the standalone
       anonymization pipeline (pattern
       ``datasets/anonymized/{dataset}_{tag}_anonymized_encoded.csv``).
    2. If absent, look for the raw anonymized CSV
       ``datasets/anonymized/{dataset}_{tag}_anonymized.csv`` and encode it.
    3. If *that* is also missing, run the Java/ARX anonymizer now and then
       encode.
    4. Iterate over every synthesizer whose config sets ``enabled: true`` and
       call :pyfunc:`run_synthetic` with a *mini config* so only that
       synthesizer is active.  Existing pickles are reused, avoiding retrain.
    5. Save each hybrid dataset to
       ``datasets/hybrid/{dataset}_{tag}_{Synth}_HYBRID.csv``.

    The function returns a ``dict[str, pd.DataFrame]`` mapping each synthesizer
    name to its hybrid dataset so downstream code (utility evaluation) can just
    consume it.
    """

    if logger:
        logger.info("Executing run_hybrid …")

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
    if os.path.exists(enc_csv):
        if logger:
            logger.info(f"Loaded cached encoded anon data to {enc_csv}")
        encoded_df = pd.read_csv(enc_csv, sep=sep)
    else:
        # ensure anonymized CSV exists first
        if not os.path.exists(anon_csv):
            if logger:
                logger.info("No anonymized CSV - running ARX anonymization …")
            success = run_anonymization(
                train_raw_path,
                config_path,
                anonymized_output_path=anon_csv,
                logger=logger,
            )
            if not success:
                raise RuntimeError("Hybrid pipeline: anonymization failed.")

        # encode once and cache
        enc_df, _, _ = run_postprocessing(anon_csv, sep, target_column, logger)
        if enc_df is None:
            raise RuntimeError("Hybrid pipeline: post-processing failed.")
        enc_df.to_csv(enc_csv, index=False, sep=sep)
        if logger:
            logger.info(f"Encoded anon data saved to {enc_csv}")
        encoded_df = enc_df

    # ---------- generate hybrid datasets ----------
    hybrid_sets: dict[str, pd.DataFrame] = {}
    for synth_name, synth_cfg in config["synthesis"]["synthesizers"].items():
        if not synth_cfg.get("enabled", False):
            continue
        if logger:
            logger.info(f"[HYBRID] Synthesizing with {synth_name} …")

        mini_cfg = copy.deepcopy(config)
        mini_cfg["synthesis"] = {
            "enable_synthetic_generation": True,
            "synthesizers": {synth_name: {**copy.deepcopy(synth_cfg), "enabled": True}},
        }

        synth_ds = run_synthetic(
            encoded_df,
            f"{dataset_name}_{anon_tag}",
            target_column,
            output_dir,
            mini_cfg,
            logger=logger,
        )

        df = synth_ds.get(synth_name)
        if df is None:
            raise RuntimeError(f"{synth_name} returned no data.")

        out_csv = os.path.join(hyb_dir, f"{dataset_name}_{anon_tag}_{synth_name}_HYBRID.csv")
        df.to_csv(out_csv, index=False, sep=sep)
        if logger:
            logger.info(f"Saved hybrid dataset to {out_csv}")
        hybrid_sets[synth_name] = df

    if logger:
        logger.info(f"Hybrid pipeline complete to {len(hybrid_sets)} dataset(s) ready.")

    return hybrid_sets
