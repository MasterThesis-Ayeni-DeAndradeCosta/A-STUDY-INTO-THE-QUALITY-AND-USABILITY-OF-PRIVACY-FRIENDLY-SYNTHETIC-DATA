# -*- coding: utf-8 -*-
"""
Auto-generate benchmark configurations for the **censusIncome** dataset.

Anonymization + Hybrid:
  - suppression_limit ∈ [0.05, 0.30, 1.0]
  - k_anonymity ∈ [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 50, 100]
  - l_diversity ∈ [1, 2]
  - Hybrid only for k ∈ {2, 6, 10, 14, 20}
"""

import itertools
from pathlib import Path
import yaml
import pandas as pd

DATASET = "censusIncome"
BASE_CFG = f"configs/base/{DATASET}_config.yaml"
if not Path(BASE_CFG).is_file():
    BASE_CFG = "configs/benchmark_config.yaml"

ORIG_DATA = f"datasets/original/{DATASET}.csv"
OUT_DIR = Path(f"configs/generated_configs/{DATASET}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_ANON = [2, 6, 10, 12, 14, 16, 18, 20, 30, 50, 75, 100, 150, 200]
SUPPRESSION = [0.05, 0.30, 1.0]
L_DIV = [1, 2]
HYBRID_ELIGIBLE_K = {6, 10, 20, 50}
HYBRID_EPOCHS = 300
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

    for sl, k, l in itertools.product(SUPPRESSION, K_ANON, L_DIV):
        cfg = load(BASE_CFG)
        cfg["dataset"]["test_size"] = TEST_SIZE_FIXED
        cfg["anonymization"].update({
            "enable_anonymization": True,
            "suppression_limit": sl,
        })
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = l

        cfg["hybrid"]["enable_hybrid"] = k in HYBRID_ELIGIBLE_K
        if cfg["hybrid"]["enable_hybrid"]:
            cfg["hybrid"]["synthesizer"] = "TVAE"

        cfg["synthesis"]["enable_synthetic_generation"] = False

        for name, s in cfg["synthesis"]["synthesizers"].items():
            s["enabled"] = name in {"TVAE", "CTGAN", "GaussianCopula"}
            if name in {"TVAE", "CTGAN"}:
                s.setdefault("params", {})["epochs"] = HYBRID_EPOCHS
            elif name == "GaussianCopula" and "params" in s:
                s["params"].pop("epochs", None)

        fname = f"{DATASET}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l{l}"
        save(cfg, fname)
        total += 1

    print(f"✅ Generated {total} configs → {OUT_DIR}")

if __name__ == "__main__":
    generate()
