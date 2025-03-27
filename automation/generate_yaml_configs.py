import os
import yaml
import pandas as pd
import itertools

# ---------- CONFIG ----------
DATASET_NAME = "loan"
MODE = "anonymization"  # or "synthesis"
BASE_CONFIG_PATH = f"configs/base/{DATASET_NAME}_config.yaml"
ORIGINAL_DATA_PATH = f"datasets/original/{DATASET_NAME}.csv"
OUTPUT_DIR = f"configs/generated_configs/{DATASET_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step


# ---------- PARAMETER SETS ----------
K_ANONYMITY = [2, 3, 5, 10, 15, 20]
L_DIVERSITY = [2, 3, 4, 5]
SUPPRESSION_LIMITS = [0.01, 0.05, 0.1]
TEST_SIZES = [round(x, 2) for x in [*frange(0.05, 0.55, 0.05)]]
EPOCHS = list(range(5, 55, 5))
ROW_MULTIPLIERS = [1, 2, 3, 5, 10]

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

# ---------- MAIN GENERATORS ----------
def generate_anonymization_configs():
    total = 0
    for k, l, sl in itertools.product(K_ANONYMITY, L_DIVERSITY, SUPPRESSION_LIMITS):
        for ts in TEST_SIZES:
            cfg = load_yaml(BASE_CONFIG_PATH)
            cfg["dataset"]["test_size"] = ts
            cfg["anonymization"]["models"]["k_anonymity"] = k
            cfg["anonymization"]["models"]["l_diversity"]["value"] = l
            cfg["anonymization"]["suppression_limit"] = sl
            cfg["synthesis"]["enable_synthetic_generation"] = False
            name = f"anon_k{k}_l{l}_sl{sl}_ts{ts}"
            save_yaml(cfg, name)
            total += 1
    return total

def generate_synthesis_configs():
    total = 0
    rows = row_count(ORIGINAL_DATA_PATH)
    for ts in TEST_SIZES:
        for ep in EPOCHS:
            for mult in ROW_MULTIPLIERS:
                gen_rows = int(rows * mult)
                cfg = load_yaml(BASE_CONFIG_PATH)
                cfg["dataset"]["test_size"] = ts
                for synth in ["CTGAN", "TVAE"]:
                    if cfg["synthesis"]["synthesizers"][synth]["enabled"]:
                        cfg["synthesis"]["synthesizers"][synth]["params"]["epochs"] = ep
                        cfg["synthesis"]["synthesizers"][synth]["custom_generated_rows"] = gen_rows
                if cfg["synthesis"]["synthesizers"]["GaussianCopula"]["enabled"]:
                    cfg["synthesis"]["synthesizers"]["GaussianCopula"]["custom_generated_rows"] = gen_rows
                name = f"synth_ts{ts}_ep{ep}_rows{mult}x"
                save_yaml(cfg, name)
                total += 1
    return total

# ---------- ENTRY ----------
def main():
    if MODE == "anonymization":
        count = generate_anonymization_configs()
    elif MODE == "synthesis":
        count = generate_synthesis_configs()
    else:
        raise ValueError("MODE must be either 'anonymization' or 'synthesis'")
    print(f"✅ {count} configs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
