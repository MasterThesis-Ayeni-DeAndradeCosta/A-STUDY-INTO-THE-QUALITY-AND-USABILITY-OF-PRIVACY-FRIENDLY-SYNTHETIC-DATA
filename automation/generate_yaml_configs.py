import os
import yaml
import pandas as pd
import itertools

# ---------- CONFIG ----------
DATASET_NAME = "crimeData"
BASE_CONFIG_PATH = f"configs/base/{DATASET_NAME}_config.yaml"
ORIGINAL_DATA_PATH = f"datasets/original/{DATASET_NAME}.csv"
OUTPUT_DIR = f"configs/generated_configs/{DATASET_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def frange(start, stop, step):
    while start < stop:
        yield round(start, 2)
        start += step

# ---------- PARAMETER SETS ----------
K_ANONYMITY = [2, 3, 5, 10, 15, 20]
L_DIVERSITY = [2, 3, 4, 5]
SUPPRESSION_LIMITS = [0.01, 0.05, 0.1]
TEST_SIZES = list(frange(0.05, 0.55, 0.05))
EPOCHS = list(range(5, 55, 5))
ROW_MULTIPLIERS = [1, 2, 3, 5, 10]

# ---------- FIXED VALUES ----------
FIXED_TEST_SIZE = 0.3
FIXED_EPOCHS = 50
FIXED_ROW_MULTIPLIER = 1

# ---------- UTILS ----------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(config, filename):
    path = os.path.join(OUTPUT_DIR, f"{filename}.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)

def row_count(path):
    return len(pd.read_csv(path))

# ---------- MAIN CONFIG GENERATOR ----------
def generate_yaml_configs():
    total = 0
    rows = row_count(ORIGINAL_DATA_PATH)

    # --- 1. Anonymization configs ---
    for k, l, sl in itertools.product(K_ANONYMITY, L_DIVERSITY, SUPPRESSION_LIMITS):
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["anonymization"]["models"]["k_anonymity"] = k
        cfg["anonymization"]["models"]["l_diversity"]["value"] = l
        cfg["anonymization"]["suppression_limit"] = sl
        cfg["synthesis"]["enable_synthetic_generation"] = False
        name = f"{DATASET_NAME}_anon_k{k}_l{l}_sl{sl}"
        save_yaml(cfg, name)
        total += 1

    # --- 2. Synthesis configs (epochs only) ---
    for ep in EPOCHS:
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = FIXED_TEST_SIZE
        gen_rows = int(rows * FIXED_ROW_MULTIPLIER)

        for synth in cfg["synthesis"]["synthesizers"]:
            if cfg["synthesis"]["synthesizers"][synth]["enabled"]:
                cfg["synthesis"]["synthesizers"][synth]["custom_generated_rows"] = gen_rows
                if "params" in cfg["synthesis"]["synthesizers"][synth]:
                    cfg["synthesis"]["synthesizers"][synth]["params"]["epochs"] = ep

        name = f"{DATASET_NAME}_synth_ep{ep}"
        save_yaml(cfg, name)
        total += 1

    # --- 3. Synthesis configs (row_multipliers only) ---
    for mult in ROW_MULTIPLIERS:
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = FIXED_TEST_SIZE
        gen_rows = int(rows * mult)

        for synth in cfg["synthesis"]["synthesizers"]:
            if cfg["synthesis"]["synthesizers"][synth]["enabled"]:
                cfg["synthesis"]["synthesizers"][synth]["custom_generated_rows"] = gen_rows
                if "params" in cfg["synthesis"]["synthesizers"][synth]:
                    cfg["synthesis"]["synthesizers"][synth]["params"]["epochs"] = FIXED_EPOCHS

        name = f"{DATASET_NAME}_synth_rows{mult}x"
        save_yaml(cfg, name)
        total += 1

    # --- 4. Base config with test_size variation ---
    for ts in TEST_SIZES:
        cfg = load_yaml(BASE_CONFIG_PATH)
        cfg["dataset"]["test_size"] = ts
        name = f"{DATASET_NAME}_testsize_{str(ts).replace('.', 'p')}"
        save_yaml(cfg, name)
        total += 1

    # ---------- VARIATION INFO GENERATION ----------
    variation_info = {
        "dataset": DATASET_NAME,
        "varied_parameters": [],
        "values": {}
    }

    if len(K_ANONYMITY) > 1:
        variation_info["varied_parameters"].append("k_anonymity")
        variation_info["values"]["k_anonymity"] = K_ANONYMITY

    if len(L_DIVERSITY) > 1:
        variation_info["varied_parameters"].append("l_diversity")
        variation_info["values"]["l_diversity"] = L_DIVERSITY

    if len(SUPPRESSION_LIMITS) > 1:
        variation_info["varied_parameters"].append("suppression_limit")
        variation_info["values"]["suppression_limit"] = SUPPRESSION_LIMITS

    if len(EPOCHS) > 1:
        variation_info["varied_parameters"].append("epochs")
        variation_info["values"]["epochs"] = EPOCHS

    if len(ROW_MULTIPLIERS) > 1:
        variation_info["varied_parameters"].append("custom_generated_rows")
        variation_info["values"]["custom_generated_rows"] = ROW_MULTIPLIERS

    if len(TEST_SIZES) > 1:
        variation_info["varied_parameters"].append("test_size")
        variation_info["values"]["test_size"] = TEST_SIZES

    variation_info_path = os.path.join(OUTPUT_DIR, "variation_info.yaml")
    with open(variation_info_path, "w") as f:
        yaml.dump(variation_info, f)

    print(f"📄 variation_info.yaml saved to: {variation_info_path}")
    print(f"✅ {total} configs saved to {OUTPUT_DIR}")

# ---------- ENTRY ----------
if __name__ == "__main__":
    generate_yaml_configs()
