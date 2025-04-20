# -*- coding: utf-8 -*-
"""
Auto-generate benchmark configurations for the **bankMarketing** dataset.

1. Anonymization + Hybrid:
   * suppression_limit ∈ [0.01, 0.03, 0.05, 0.08, 0.10, 0.25, 0.50]
   * k_anonymity ∈ [5, 10, 20] (l_diversity = 2)
   * Hybrid enabled with CTGAN, TVAE (epochs = 300), GaussianCopula

2. Synthetic-only:
   * epochs ∈ [50,100,200,300,500,750,1000]
   * row_multiplier ∈ [1, 2, 3, 5, 10]
   * GaussianCopula also tested at epoch == 300 using same multiplier

3. Evaluation baseline:
   * Fixed test_size = 0.20 with both anonymization & synthesis enabled
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
HYBRID_EPOCHS = 300

EPOCHS = [50, 100, 200, 300, 500, 750, 1000]
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

    # ---- 1) Anonymization + Hybrid ---------------------------------------
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

        for name, s in cfg["synthesis"]["synthesizers"].items():
            s["enabled"] = name in {"TVAE", "CTGAN", "GaussianCopula"}
            if name in {"TVAE", "CTGAN"}:
                s.setdefault("params", {})["epochs"] = HYBRID_EPOCHS
            elif name == "GaussianCopula" and "params" in s:
                s["params"].pop("epochs", None)  # ✅ Only remove epochs

        fname = f"{DATASET}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l2"
        save(cfg, fname)
        total += 1

    # ---- 2) Synthetic-only -----------------------------------------------
    for ep, mult in itertools.product(EPOCHS, MULTIPLIERS):
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = TEST_SIZE_FIXED
        cfg["anonymization"]["enable_anonymization"] = False
        cfg["synthesis"]["enable_synthetic_generation"] = True

        for name, s in cfg["synthesis"]["synthesizers"].items():
            if name in {"CTGAN", "TVAE"}:
                s["enabled"] = True
                s.setdefault("params", {})["epochs"] = ep
                s["num_generated_rows"] = "multiple"
                s["row_multiplier"] = mult
                s.pop("custom_generated_rows", None)

            elif name == "GaussianCopula":
                if ep == 300:
                    s["enabled"] = True
                    s["num_generated_rows"] = "multiple"
                    s["row_multiplier"] = mult
                    if "params" in s:
                        s["params"].pop("epochs", None)
                    s.pop("custom_generated_rows", None)
                else:
                    s["enabled"] = False
            else:
                s["enabled"] = False

        fname = f"{DATASET}_synth_ep{ep}_rows{mult}x"
        save(cfg, fname)
        total += 1

    # ---- 3) Evaluation baseline ------------------------------------------
    base_cfg = load(BASE_CFG)
    base_cfg["dataset"]["test_size"] = TEST_SIZE_FIXED
    base_cfg["anonymization"]["enable_anonymization"] = True
    base_cfg["synthesis"]["enable_synthetic_generation"] = True
    fname = f"{DATASET}_testsize_02"
    save(base_cfg, fname)
    total += 1

    # ---- variation_info ---------------------------------------------------
    info = {
        "dataset": DATASET,
        "varied_parameters": ["suppression_limit", "k_anonymity", "epochs", "row_multiplier"],
        "values": {
            "suppression_limit": SUPPRESSION,
            "k_anonymity": K_ANON,
            "epochs": EPOCHS,
            "row_multiplier": MULTIPLIERS,
        },
    }
    with open(OUT_DIR / "variation_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(info, f)

    print(f"✅ Generated {total} configs (expected 47) → {OUT_DIR}")

if __name__ == "__main__":
    generate()
