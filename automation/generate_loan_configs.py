# -*- coding: utf-8 -*-
"""
Auto-generate benchmark configurations for the **loan** dataset.

Includes:
1. Anonymization + Hybrid (44 configs)
2. Synthetic-only (65 configs)
3. Test size sweep (11 configs)
"""

import itertools
from pathlib import Path
import yaml
import pandas as pd

DATASET = "loan"
BASE_CFG = f"configs/base/{DATASET}_config.yaml"
if not Path(BASE_CFG).exists():
    BASE_CFG = "configs/benchmark_config.yaml"

ORIG_DATA = f"datasets/original/{DATASET}.csv"
OUT_DIR = Path(f"configs/generated_configs/{DATASET}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPRESSION_LIMITS = [0.01, 0.05, 0.10, 0.20, 0.50, 1]
K_ANONYMITY = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 50]
L_DIVERSITY = [2, 5, 10, 15]
EPOCHS = [20, 40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 1000]
ROW_MULTIPLIERS = [1, 2, 3, 5, 10]
TEST_SIZES = [round(x * 0.05, 2) for x in range(1, 12)]
DEFAULT_TEST_SIZE = 0.30
HYBRID_EPOCHS = 300

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save(cfg, name):
    with open(OUT_DIR / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

def n_rows(path):
    return len(pd.read_csv(path))

def generate():
    total = 0
    rows = n_rows(ORIG_DATA)

    # ---- 1. Anonymization + Hybrid (fixed test size) ----
    for sl, k, l in itertools.product(SUPPRESSION_LIMITS, K_ANONYMITY, L_DIVERSITY):
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = DEFAULT_TEST_SIZE
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["anonymization"]["suppression_limit"] = sl
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = l

        cfg["hybrid"]["enable_hybrid"] = True
        cfg["hybrid"]["synthesizer"] = "TVAE"
        cfg["synthesis"]["enable_synthetic_generation"] = False

        for name, s in cfg["synthesis"]["synthesizers"].items():
            s["enabled"] = name in {"TVAE", "CTGAN", "GaussianCopula"}
            if name in {"TVAE", "CTGAN"}:
                s.setdefault("params", {})["epochs"] = HYBRID_EPOCHS
            elif name == "GaussianCopula":
                s.pop("params", None)

        fname = f"{DATASET}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l{l}"
        save(cfg, fname)
        total += 1

    # ---- 2. Synthetic-only (fixed test size) ----
    for ep, mult in itertools.product(EPOCHS, ROW_MULTIPLIERS):
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = DEFAULT_TEST_SIZE
        cfg["anonymization"]["enable_anonymization"] = False
        cfg["hybrid"]["enable_hybrid"] = False
        cfg["synthesis"]["enable_synthetic_generation"] = True

        for name, s in cfg["synthesis"]["synthesizers"].items():
            if name in {"CTGAN", "TVAE"}:
                s["enabled"] = True
                s.setdefault("params", {})["epochs"] = ep
                s["num_generated_rows"] = "multiple"
                s["row_multiplier"] = mult
                s.pop("custom_generated_rows", None)
            elif name == "GaussianCopula":
                s["enabled"] = (ep == 300)
                if ep == 300:
                    s["num_generated_rows"] = "multiple"
                    s["row_multiplier"] = mult
                    s.pop("custom_generated_rows", None)
                s.pop("params", None)

        fname = f"{DATASET}_synth_ep{ep}_rows{mult}x"
        save(cfg, fname)
        total += 1

    # ---- 3. Test size sweep (only test_size changes) ----
    for ts in TEST_SIZES:
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = ts
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["synthesis"]["enable_synthetic_generation"] = True
        fname = f"{DATASET}_testsize_{str(ts).replace('.', 'p')}"
        save(cfg, fname)
        total += 1

    # ---- variation_info.yaml ----
    info = {
        "dataset": DATASET,
        "varied_parameters": ["suppression_limit", "k_anonymity", "epochs", "row_multiplier", "test_size"],
        "values": {
            "suppression_limit": SUPPRESSION_LIMITS,
            "k_anonymity": K_ANONYMITY,
            "epochs": EPOCHS,
            "row_multiplier": ROW_MULTIPLIERS,
            "test_size": TEST_SIZES,
        },
    }
    with open(OUT_DIR / "variation_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(info, f)

    print(f"✅ Generated {total} configs → {OUT_DIR}")

if __name__ == "__main__":
    generate()
