# -*- coding: utf-8 -*-
"""
Auto-generate 120 YAML benchmark configurations for the **loan** dataset.

1. Anonymization + Hybrid (44 cfgs)
   * suppression_limit ∈ [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50]
   * k_anonymity ∈ [5, 10, 15, 20]  (l_diversity = 2)
   * CTGAN, TVAE (300 epochs), and GaussianCopula enabled for hybrid.

2. Synthetic generation only (65 cfgs)
   * epochs ∈ [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 1000]
   * row multipliers ∈ [1, 2, 3, 5, 10]
   * CTGAN & TVAE always enabled.
   * GaussianCopula enabled only when epochs == 300.

3. Evaluation test size sweep (11 cfgs)
   * test_size ∈ [0.05, ..., 0.55]

Outputs → configs/generated_configs/loan/
"""

import os
import itertools
import yaml
import pandas as pd
from pathlib import Path

DATASET_NAME = "loan"
BASE_CONFIG_PATH = f"configs/base/{DATASET_NAME}_config.yaml"
if not Path(BASE_CONFIG_PATH).is_file():
    BASE_CONFIG_PATH = "configs/benchmark_config.yaml"

ORIGINAL_DATA_PATH = f"datasets/original/{DATASET_NAME}.csv"
OUTPUT_DIR = Path(f"configs/generated_configs/{DATASET_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPRESSION_LIMITS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50]
K_ANONYMITY = [5, 10, 15, 20]
L_DIVERSITY = 2

EPOCHS = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 1000]
ROW_MULTIPLIERS = [1, 2, 3, 5, 10]
TEST_SIZES = [round(x * 0.05, 2) for x in range(1, 12)]  # 0.05 to 0.55

DEFAULT_TEST_SIZE = 0.30
HYBRID_EPOCHS = 300

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(cfg: dict, filename: str):
    with open(OUTPUT_DIR / f"{filename}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

def row_count(path):
    return len(pd.read_csv(path))

def generate_yaml_configs():
    total = 0
    n_rows = row_count(ORIGINAL_DATA_PATH)

    # ---- Anonymization + Hybrid (44 configs)
    for sl, k in itertools.product(SUPPRESSION_LIMITS, K_ANONYMITY):
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = DEFAULT_TEST_SIZE

        cfg["anonymization"]["enable_anonymization"] = True
        cfg["anonymization"]["suppression_limit"] = sl
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = L_DIVERSITY

        cfg["hybrid"]["enable_hybrid"] = True
        cfg["synthesis"]["enable_synthetic_generation"] = False

        for name, synth in cfg["synthesis"]["synthesizers"].items():
            synth["enabled"] = name in {"CTGAN", "TVAE", "GaussianCopula"}
            if name in {"CTGAN", "TVAE"}:
                synth.setdefault("params", {})["epochs"] = HYBRID_EPOCHS

        fname = f"{DATASET_NAME}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l2"
        save_yaml(cfg, fname)
        total += 1

    # ---- Synthetic-only (65 configs)
    for ep, mult in itertools.product(EPOCHS, ROW_MULTIPLIERS):
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = DEFAULT_TEST_SIZE
        cfg["anonymization"]["enable_anonymization"] = False
        cfg["synthesis"]["enable_synthetic_generation"] = True

        gen_rows = int(n_rows * mult)

        for name, synth in cfg["synthesis"]["synthesizers"].items():
            synth["enabled"] = False

        for name in ("CTGAN", "TVAE"):
            synth = cfg["synthesis"]["synthesizers"][name]
            synth["enabled"] = True
            synth.setdefault("params", {})["epochs"] = ep
            synth["num_generated_rows"] = "custom"
            synth["custom_generated_rows"] = gen_rows

        if ep == 300:
            gc = cfg["synthesis"]["synthesizers"]["GaussianCopula"]
            gc["enabled"] = True
            gc["num_generated_rows"] = "custom"
            gc["custom_generated_rows"] = gen_rows

        fname = f"{DATASET_NAME}_synth_ep{ep}_rows{mult}x"
        save_yaml(cfg, fname)
        total += 1

    # ---- Test Size Sweep (11 configs)
    for ts in TEST_SIZES:
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = ts
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["synthesis"]["enable_synthetic_generation"] = True
        fname = f"{DATASET_NAME}_testsize_{str(ts).replace('.', 'p')}"
        save_yaml(cfg, fname)
        total += 1

    # ---- variation_info.yaml
    variation_info = {
        "dataset": DATASET_NAME,
        "varied_parameters": [
            "suppression_limit",
            "k_anonymity",
            "epochs",
            "custom_generated_rows",
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

    with open(OUTPUT_DIR / "variation_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(variation_info, f)

    print(f"✅ Generated {total} configs → {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_yaml_configs()