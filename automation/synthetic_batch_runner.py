import itertools
import copy
import yaml
import subprocess
import datetime
import sys
import os

# Dynamically add 'src/' to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from analysis.batch_analysis import analyze_batch_results

# ----------- Config Paths -----------
BASE_CONFIG_PATH = "configs/benchmark_config.yaml"
TEMP_CONFIG_PATH = "configs/temp_config.yaml"
BATCH_OUTPUT_DIR = os.path.join("outputs", "batch")
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

# ----------- Parameters to Vary -----------
PARAMETER_VARIATIONS = [
    {
        "path": "ALL_SYNTHS.epochs",
        "values": [1, 2, 3, 4, 5]
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
    # Save parameter variation info in batch folder
    variation_info = {
        "varied_parameter": param_paths[0].split(".")[-1],  # e.g., "epochs"
        "values": param_values_list[0]
    }
    variation_info_path = os.path.join(master_batch_folder_path, "variation_info.yaml")
    with open(variation_info_path, 'w') as f:
        yaml.dump(variation_info, f)


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
    analyze_batch_results(master_batch_folder_path)

# def analyze_batch_results(master_batch_folder_path):
#     import seaborn as sns

#     # Load parameter info
#     variation_info_path = os.path.join(master_batch_folder_path, "variation_info.yaml")
#     with open(variation_info_path, 'r') as f:
#         variation_info = yaml.safe_load(f)
#     param_name = variation_info["varied_parameter"]

#     all_dfs = []

#     # Gather all result CSVs and inject param value
#     for run_folder in os.listdir(master_batch_folder_path):
#         run_path = os.path.join(master_batch_folder_path, run_folder)

#         if not os.path.isdir(run_path):
#             continue

#         result_csv = os.path.join(run_path, "model_performance.csv")
#         if os.path.exists(result_csv):
#             df = pd.read_csv(result_csv)
#             try:
#                 import re
#                 match = re.search(rf"{param_name}(\d+)", run_folder)
#                 if match:
#                     param_value = int(match.group(1))
#                     df[param_name] = param_value
#                     all_dfs.append(df)
#             except ValueError:
#                 continue

#     if not all_dfs:
#         print("❌ No result CSVs found.")
#         return

#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     combined_df.to_csv(os.path.join(master_batch_folder_path, "combined_results.csv"), index=False)

#     # 🔍 Dynamically detect metric columns (exclude known non-metrics)
#     non_metrics = {"Model", "Dataset", param_name}
#     metric_columns = [col for col in combined_df.columns if col not in non_metrics and pd.api.types.is_numeric_dtype(combined_df[col])]

#     print(f"📊 Metrics detected: {metric_columns}")

#     for metric in metric_columns:
#         plt.figure(figsize=(10, 6))
#         sns.lineplot(data=combined_df, x=param_name, y=metric,
#                      hue="Dataset", style="Model", marker="o")

#         plt.title(f"{metric} vs {param_name.capitalize()} (All Synthesizers + Models)")
#         plt.grid(True)
#         plt.tight_layout()

#         plot_path = os.path.join(master_batch_folder_path, f"{metric.lower()}_vs_{param_name}.png")
#         plt.savefig(plot_path, dpi=300)
#         plt.close()
#         print(f"✅ Saved plot: {plot_path}")

#     print("\n🎯 Analysis complete for all metrics.")


# ----------- Execute When Run Directly -----------
if __name__ == "__main__":
    run_batch_variations()
