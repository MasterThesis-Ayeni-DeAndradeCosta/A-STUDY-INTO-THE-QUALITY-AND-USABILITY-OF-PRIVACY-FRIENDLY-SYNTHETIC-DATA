import itertools
import copy
import yaml
import subprocess
import datetime
import os

# ----------- Config Paths -----------
BASE_CONFIG_PATH = "configs/benchmark_config.yaml"
TEMP_CONFIG_PATH = "configs/temp_config.yaml"
BATCH_OUTPUT_DIR = os.path.join("outputs", "batch")
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

# ----------- Parameters to Vary -----------
PARAMETER_VARIATIONS = [
    {
        "path": "ALL_SYNTHS.epochs",
        "values": [1, 2, 3]
    }
]

# ----------- Utility Functions -----------

def set_nested_value(config, path, value):
    keys = path.split(".")
    sub_config = config
    for key in keys[:-1]:
        sub_config = sub_config[key]
    sub_config[keys[-1]] = value

def generate_output_folder_name(param_combo):
    parts = []
    for key, value in param_combo.items():
        short_key = key.split(".")[-1] if "." in key else key
        val_str = str(value).replace('.', 'p')
        parts.append(f"{short_key}{val_str}")
    return "_".join(parts)

# ----------- Batch Runner Function -----------

def run_batch_variations():
    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    dataset_path = base_config["dataset"]["path"]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    # Create master batch folder
    batch_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    master_batch_folder_name = f"{dataset_name}_batch_{batch_timestamp}"
    master_batch_folder_path = os.path.join(BATCH_OUTPUT_DIR, master_batch_folder_name)
    os.makedirs(master_batch_folder_path, exist_ok=True)

    print(f"\n📁 Master batch folder: {master_batch_folder_path}")

    # Prepare parameter combinations
    param_paths = [item["path"] for item in PARAMETER_VARIATIONS]
    param_values_list = [item["values"] for item in PARAMETER_VARIATIONS]
    param_combinations = list(itertools.product(*param_values_list))

    for combo in param_combinations:
        config = copy.deepcopy(base_config)
        param_combo_dict = dict(zip(param_paths, combo))

        # Apply parameter values
        for path, value in param_combo_dict.items():
            if path == "ALL_SYNTHS.epochs":
                for synth_name, synth_info in config["synthesis"]["synthesizers"].items():
                    if not synth_info.get("enabled", False):
                        continue
                    params = synth_info.get("params", {})
                    if "epochs" in params:
                        params["epochs"] = value
            else:
                set_nested_value(config, path, value)

        # Save temp config
        with open(TEMP_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f)

        # Create subfolder for this run inside master folder
        folder_name = generate_output_folder_name(param_combo_dict)
        run_output_path = os.path.join(master_batch_folder_path, folder_name)
        os.makedirs(run_output_path, exist_ok=True)

        # Inject this run’s output path into config
        config["output_dir"] = run_output_path
        with open(TEMP_CONFIG_PATH, 'w') as f:
            yaml.dump(config, f)

        print(f"\n▶ Running with parameters: {param_combo_dict}")
        print(f"Output for this run: {run_output_path}")

        subprocess.run(["python", "scripts/run_benchmarks.py", "--config", TEMP_CONFIG_PATH])

    print(f"\n✅ All batch runs completed. Master folder: {master_batch_folder_path}")

# ----------- Execute When Run Directly -----------
if __name__ == "__main__":
    run_batch_variations()
