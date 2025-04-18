# -*- coding: utf-8 -*-
"""
Auto-generate 47 YAML benchmark configurations for the **bankMarketing**
dataset (45 211 rows, 17 columns) according to the April 17 2025 plan:

1. **Anonymization + Hybrid** (21 cfgs)
   * suppression_limit ∈ [0.01, 0.03, 0.05, 0.08, 0.10, 0.25, 0.50]
   * k_anonymity     ∈ [5, 10, 20]   (l_diversity = 2)
   * Hybrid enabled with **TVAE** (epochs = 60).  No stand-alone synthesis.

2. **Synthetic- only** (25 cfgs)
   * epochs ∈ [20, 40, 60, 80, 100]
   * row_multipliers ∈ [1, 2, 3, 5, 10]  → custom_generated_rows.
   * CTGAN & TVAE always enabled; no Gaussian Copula in this grid.

3. **Evaluation baseline** (1 cfg)
   * Fixed test_size = 0.20 with both anonymization & synthesis enabled.

Outputs land in `configs/generated_configs/bankMarketing/` plus
`variation_info.yaml`.
"""

import itertools
from pathlib import Path
import yaml
import pandas as pd

DATASET = "bankMarketing"
BASE_CFG = f"configs/base/{DATASET}_config.yaml"
if not Path(BASE_CFG).is_file():
    BASE_CFG = "configs/benchmark_config.yaml"

ORIG_DATA = f"datasets/original/{DATASET}.csv"
OUT_DIR = Path(f"configs/generated_configs/{DATASET}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPRESSION = [0.01, 0.03, 0.05, 0.08, 0.10, 0.25, 0.50]
K_ANON = [5, 10, 20]
L_DIV = 2
HYBRID_EPOCHS = 100

EPOCHS = [20, 40, 60, 80, 100]
MULTIPLIERS = [1, 2, 3, 5, 10]
TEST_SIZE_FIXED = 0.20

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save(cfg: dict, name: str):
    with open(OUT_DIR / f"{name}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

def n_rows(path):
    return len(pd.read_csv(path))

def generate():
    total = 0
    rows = n_rows(ORIG_DATA)

    # ---- 1) Anonymization + Hybrid ------------------------------------
    for sl, k in itertools.product(SUPPRESSION, K_ANON):
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = TEST_SIZE_FIXED
        cfg["anonymization"].update({
            "enable_anonymization": True,
            "suppression_limit": sl,
        })
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = L_DIV

        cfg["hybrid"].update({"enable_hybrid": True, "synthesizer": "TVAE"})
        cfg["synthesis"]["enable_synthetic_generation"] = False

        # Enable all 3 synthesizers for hybrid reuse
        for name, s in cfg["synthesis"]["synthesizers"].items():
            s["enabled"] = name in {"TVAE", "CTGAN", "GaussianCopula"}
            if name in {"TVAE", "CTGAN"}:
                s.setdefault("params", {})["epochs"] = HYBRID_EPOCHS

        fname = f"{DATASET}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l2"
        save(cfg, fname)
        total += 1

    # ---- 2) Synthetic-only --------------------------------------------
    for ep, mult in itertools.product(EPOCHS, MULTIPLIERS):
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = TEST_SIZE_FIXED
        cfg["anonymization"]["enable_anonymization"] = False
        cfg["synthesis"]["enable_synthetic_generation"] = True

        gen_rows = int(rows * mult)
        for name, s in cfg["synthesis"]["synthesizers"].items():
            if name in {"CTGAN", "TVAE"}:
                s["enabled"] = True
                s.setdefault("params", {})["epochs"] = ep
                s["num_generated_rows"] = "custom"
                s["custom_generated_rows"] = gen_rows
            else:
                s["enabled"] = False  # GaussianCopula excluded

        fname = f"{DATASET}_synth_ep{ep}_rows{mult}x"
        save(cfg, fname)
        total += 1

    # ---- 3) Evaluation baseline ---------------------------------------
    base_cfg = load(BASE_CFG)
    base_cfg["dataset"]["test_size"] = TEST_SIZE_FIXED
    base_cfg["anonymization"]["enable_anonymization"] = True
    base_cfg["synthesis"]["enable_synthetic_generation"] = True
    fname = f"{DATASET}_testsize_02"
    save(base_cfg, fname)
    total += 1

    # ---- variation_info -----------------------------------------------
    info = {
        "dataset": DATASET,
        "varied_parameters": ["suppression_limit", "k_anonymity", "epochs", "custom_generated_rows"],
        "values": {
            "suppression_limit": SUPPRESSION,
            "k_anonymity": K_ANON,
            "epochs": EPOCHS,
            "custom_generated_rows": MULTIPLIERS,
        },
    }
    with open(OUT_DIR / "variation_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(info, f)

    print(f"✅ Generated {total} configs (expected 47) → {OUT_DIR}")

if __name__ == "__main__":
    generate()