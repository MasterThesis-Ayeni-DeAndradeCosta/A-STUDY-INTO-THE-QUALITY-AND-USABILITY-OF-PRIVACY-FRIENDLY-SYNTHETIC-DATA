# -*- coding: utf-8 -*-
"""
Generate 82 YAML benchmark configurations for the **studentPerformance**
dataset, matching the revised experiment grid (Apr 17 2025).

Configuration counts
--------------------
1. Anonymization + Hybrid  : 9 suppression x 4 k = 36
2. Stand-alone Synthetic   : 7 epochs x 5 multipliers = 35
3. Test-size evaluation    : 11 sizes                = 11
-------------------------------------------------------
Total                                            = 82

Key rules
~~~~~~~~~
* **Anonymization/Hybrid**
  * suppression_limit ∈ [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.20, 0.30]
  * k_anonymity       ∈ [2, 5, 10, 20]  (l_diversity fixed to 2)
  * Hybrid enabled with **CTGAN** only (epochs = 100).  Synthesis disabled.
* **Synthetic-only**
  * epochs ∈ [20, 40, 60, 80, 100, 120, 150]
  * row multipliers ∈ [1, 2, 3, 5, 10]
  * CTGAN & TVAE always enabled.  Gaussian Copula additionally enabled when
    epochs == 100 (same multiplier).
* **Evaluation sweep**
  * test_size ∈ 0.05 … 0.55, step 0.05.

Outputs are written to `configs/generated_configs/studentPerformance/` with a
`variation_info.yaml` describing all varied parameters.
"""

from __future__ import annotations
import itertools
import os
from pathlib import Path
import yaml
import pandas as pd

# ---------------------------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------------------------
DATASET_NAME = "studentPerformance"
BASE_CONFIG_PATH = f"configs/base/{DATASET_NAME}_config.yaml"
if not Path(BASE_CONFIG_PATH).is_file():
    BASE_CONFIG_PATH = "configs/benchmark_config.yaml"  # fallback

ORIG_DATA_PATH = f"datasets/original/{DATASET_NAME}.csv"
OUT_DIR = Path(f"configs/generated_configs/{DATASET_NAME}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- parameter grids ------------------------------------------------------
SUPPRESSION_LIMITS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.20, 0.30]
K_ANONYMITY = [2, 5, 10, 20]
L_DIVERSITY = 2

EPOCHS = [20, 40, 60, 80, 100, 120, 150]
ROW_MULTIPLIERS = [1, 2, 3, 5, 10]

TEST_SIZES = [round(x * 0.05, 2) for x in range(1, 12)]  # 0.05 … 0.55

# ---- defaults -------------------------------------------------------------
DEFAULT_TS = 0.30  # used when test_size is not varied
HYBRID_EPOCHS = 100  # CTGAN epochs for hybrid stage

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(cfg: dict, name: str):
    path = OUT_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)


def n_rows(path: str | Path) -> int:
    return len(pd.read_csv(path))

# ---------------------------------------------------------------------------
#  generator
# ---------------------------------------------------------------------------

def generate_configs():
    total = 0
    rows = n_rows(ORIG_DATA_PATH)

    # ---- 1) anonymization + hybrid ---------------------------------------
    for sl, k in itertools.product(SUPPRESSION_LIMITS, K_ANONYMITY):
        cfg = load_yaml(BASE_CONFIG_PATH)

        cfg["dataset"]["test_size"] = DEFAULT_TS
        # anonymization params
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["anonymization"]["suppression_limit"] = sl
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = L_DIVERSITY

        # hybrid enabled with CTGAN only
        cfg["hybrid"]["enable_hybrid"] = True
        cfg["hybrid"]["synthesizer"] = "CTGAN"
        cfg["synthesis"]["enable_synthetic_generation"] = False

        # adjust synthesizer block so CTGAN is ready for reuse
        for name, s in cfg["synthesis"]["synthesizers"].items():
            s["enabled"] = (name == "CTGAN")
            if name == "CTGAN":
                s.setdefault("params", {})["epochs"] = HYBRID_EPOCHS

        fname = f"{DATASET_NAME}_anon_k{k}_sl{str(sl).replace('.', 'p')}_l2"
        save_yaml(cfg, fname)
        total += 1

    # ---- 2) synthetic‑only ------------------------------------------------
    for ep, mult in itertools.product(EPOCHS, ROW_MULTIPLIERS):
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = DEFAULT_TS
        cfg["anonymization"]["enable_anonymization"] = False
        cfg["synthesis"]["enable_synthetic_generation"] = True

        gen_rows = int(rows * mult)

        # reset synth flags
        for s in cfg["synthesis"]["synthesizers"].values():
            s["enabled"] = False

        # enable CTGAN & TVAE
        for name in ("CTGAN", "TVAE"):
            synth = cfg["synthesis"]["synthesizers"][name]
            synth["enabled"] = True
            synth.setdefault("params", {})["epochs"] = ep
            synth["num_generated_rows"] = "custom"
            synth["custom_generated_rows"] = gen_rows

        # Gaussian Copula only when ep == 100
        if ep == 100:
            gc = cfg["synthesis"]["synthesizers"]["GaussianCopula"]
            gc["enabled"] = True
            gc["num_generated_rows"] = "custom"
            gc["custom_generated_rows"] = gen_rows

        fname = f"{DATASET_NAME}_synth_ep{ep}_rows{mult}x"
        save_yaml(cfg, fname)
        total += 1

    # ---- 3) test_size sweep ----------------------------------------------
    for ts in TEST_SIZES:
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = ts
        cfg["anonymization"]["enable_anonymization"] = True
        cfg["synthesis"]["enable_synthetic_generation"] = True

        fname = f"{DATASET_NAME}_testsize_{str(ts).replace('.', 'p')}"
        save_yaml(cfg, fname)
        total += 1

    # ---- variation_info.yaml ---------------------------------------------
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

    with open(OUT_DIR / "variation_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(variation_info, f)

    print(f"📄 variation_info.yaml saved to: {OUT_DIR/'variation_info.yaml'}")
    print(f"✅ Generated {total} configs in {OUT_DIR} (expected 82)")


if __name__ == "__main__":
    generate_configs()

