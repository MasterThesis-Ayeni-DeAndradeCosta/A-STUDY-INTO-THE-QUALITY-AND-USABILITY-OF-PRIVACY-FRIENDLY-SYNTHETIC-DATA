# -*- coding: utf-8 -*-
"""
Auto-generate 90 YAML benchmark configurations for the **loan** dataset.

The script mirrors the style of `generate_yaml_configs.py`, but it follows
the exact experiment plan requested on April 17 2025:

1. **Anonymization + Hybrid (44 cfgs)**
   * suppression_limit ∈ [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50]
   * k_anonymity ∈ [5, 10, 15, 20]  (l_diversity fixed to 2)
   * hybrid enabled, anonymization enabled, standalone synthesis disabled.
   * CTGAN / TVAE (epochs = 100) and Gaussian Copula *enabled* for the
     hybrid stage so they run together, but their parameters do **not**
     multiply the count (hence 11 x 4 x 1 = 44).

2. **Synthetic generation only (35 cfgs)**
   * epochs ∈ [20, 40, 60, 80, 100, 120, 150]
   * row multipliers ∈ [1, 2, 3, 5, 10]
   * CTGAN & TVAE always enabled with the selected epoch & multiplier.
   * Gaussian Copula is additionally enabled *only* when epochs == 100,
     using the same multiplier.  This yields 7 x 5 = 35 total.
   * anonymization disabled.

3. **Evaluation parameter sweep (11 cfgs)**
   * test_size ∈ [0.05 … 0.55 step 0.05]
   * both anonymization and synthesis enabled (matches the template's
     behaviour for test - size sweeps).
s
Output files go to `configs/generated_configs/loan/`, plus a
`variation_info.yaml` describing every varied parameter.
"""

import os
import itertools
import yaml
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
#  CONFIG CONSTANTS
# ---------------------------------------------------------------------------
DATASET_NAME = "loan"
BASE_CONFIG_PATH = f"configs/base/{DATASET_NAME}_config.yaml"
# fallback: if a dataset‑specific base config is missing, use the repo default
if not Path(BASE_CONFIG_PATH).is_file():
    BASE_CONFIG_PATH = "configs/benchmark_config.yaml"

ORIGINAL_DATA_PATH = f"datasets/original/{DATASET_NAME}.csv"
OUTPUT_DIR = Path(f"configs/generated_configs/{DATASET_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------  PARAMETER GRIDS  --------------------------------------
SUPPRESSION_LIMITS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50]
K_ANONYMITY = [5, 10, 15, 20]
L_DIVERSITY = 2  # fixed

EPOCHS = [20, 40, 60, 80, 100, 120, 150]
ROW_MULTIPLIERS = [1, 2, 3, 5, 10]

def frange(start: float, stop: float, step: float):
    """Yield floats with a given step (rounded to 2 decimals)."""
    while start <= stop + 1e-9:
        yield round(start, 2)
        start += step

TEST_SIZES = list(frange(0.05, 0.55, 0.05))  # 0.05 … 0.55

# --------------  FIXED VALUES FOR CROSS‑CATEGORY CONFIGS  ------------------
FIXED_TEST_SIZE = 0.30  # default split when not varied
DEFAULT_HYBRID_EPOCHS = 100  # TVAE epochs for hybrid stage

# ---------------------------------------------------------------------------
#  UTILS
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path, "r", encoding="utf‑8") as f:
        return yaml.safe_load(f)

def save_yaml(cfg: dict, filename: str):
    path = OUTPUT_DIR / f"{filename}.yaml"
    with open(path, "w", encoding="utf‑8") as f:
        yaml.dump(cfg, f)

def row_count(path):
    return len(pd.read_csv(path))

# ---------------------------------------------------------------------------
#  MAIN GENERATOR
# ---------------------------------------------------------------------------

def generate_yaml_configs():
    total = 0
    n_rows = row_count(ORIGINAL_DATA_PATH)

    # ------- 1.  ANONYMIZATION + HYBRID ------------------------------------
    for sl, k in itertools.product(SUPPRESSION_LIMITS, K_ANONYMITY):
        cfg = load_yaml(BASE_CONFIG_PATH)

        # Anonymization settings
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["anonymization"]["suppression_limit"] = sl
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = L_DIVERSITY

        # Hybrid enabled, standalone synthesis disabled
        cfg["hybrid"]["enable_hybrid"] = True
        cfg["synthesis"]["enable_synthetic_generation"] = False

        # Ensure CTGAN, TVAE (epochs = 100) & Gaussian are ready for the
        # hybrid stage – they will be reused, not stand‑alone.
        for name, synth in cfg["synthesis"]["synthesizers"].items():
            synth["enabled"] = name in {"CTGAN", "TVAE", "GaussianCopula"}
            if name in {"CTGAN", "TVAE"}:
                synth.setdefault("params", {})["epochs"] = DEFAULT_HYBRID_EPOCHS

        fname = f"{DATASET_NAME}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l2"
        save_yaml(cfg, fname)
        total += 1

    # ------- 2.  PURE SYNTHETIC GENERATION ---------------------------------
    for ep, mult in itertools.product(EPOCHS, ROW_MULTIPLIERS):
        cfg = load_yaml(BASE_CONFIG_PATH)

        cfg["dataset"]["test_size"] = FIXED_TEST_SIZE
        cfg["anonymization"]["enable_anonymization"] = False
        cfg["synthesis"]["enable_synthetic_generation"] = True

        gen_rows = int(n_rows * mult)

        for name, synth in cfg["synthesis"]["synthesizers"].items():
            # Reset all
            synth["enabled"] = False

        # --- CTGAN & TVAE ---
        for name in ("CTGAN", "TVAE"):
            synth = cfg["synthesis"]["synthesizers"][name]
            synth["enabled"] = True
            synth.setdefault("params", {})["epochs"] = ep
            synth["num_generated_rows"] = "custom"
            synth["custom_generated_rows"] = gen_rows

        # --- Gaussian Copula only when ep == 100 ---
        if ep == 100:
            gc = cfg["synthesis"]["synthesizers"]["GaussianCopula"]
            gc["enabled"] = True
            gc["num_generated_rows"] = "custom"
            gc["custom_generated_rows"] = gen_rows

        fname = f"{DATASET_NAME}_synth_ep{ep}_rows{mult}x"
        save_yaml(cfg, fname)
        total += 1

    # ------- 3.  TEST SIZE VARIATION ---------------------------------------
    for ts in TEST_SIZES:
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = ts
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["synthesis"]["enable_synthetic_generation"] = True

        fname = f"{DATASET_NAME}_testsize_{str(ts).replace('.', 'p')}"
        save_yaml(cfg, fname)
        total += 1

    # ------- 4.  variation_info.yaml ---------------------------------------
    variation_info = {
        "dataset": DATASET_NAME,
        "varied_parameters": [
            "suppression_limit",
            "k_anonymity",
            "epochs",  # for CTGAN/TVAE
            "custom_generated_rows",  # via multiplier
            "test_size",
        ],
        "values": {
            "suppression_limit": SUPPRESSION_LIMITS,
            "k_anonymity": K_ANONYMITY,
            "epochs": EPOCHS,
            "custom_generated_rows": ROW_MULTIPLIERS,
            "test_size": TEST_SIZES,
        },
    }

    with open(OUTPUT_DIR / "variation_info.yaml", "w", encoding="utf‑8") as f:
        yaml.dump(variation_info, f)

    print(f"📄 variation_info.yaml saved to: {OUTPUT_DIR / 'variation_info.yaml'}")
    print(f"✅ {total} configs generated in {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
#  ENTRY‑POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    generate_yaml_configs()
