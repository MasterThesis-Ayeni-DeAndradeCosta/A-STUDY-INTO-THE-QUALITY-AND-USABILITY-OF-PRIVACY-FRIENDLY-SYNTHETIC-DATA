import os
import subprocess
import datetime
import csv
import yaml
import shutil

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from analysis.batch_analysis import analyze_batch_results
from analysis.analyze_from_yaml_configs import analyze_batch_results_from_configs

# ---------------- HELPERS ----------------

def get_config_paths(dataset_name):
    """Returns a list of all YAML config paths for a given dataset."""
    config_dir = f"configs/generated_configs/{dataset_name}"
    return [
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith(".yaml")  and f != "variation_info.yaml"
    ]

def create_batch_folder(dataset_name):
    """Creates and returns a timestamped output folder for the batch run."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join("outputs", "batch", f"{dataset_name}_batch_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder

def run_config(config_path, output_path):
    # Inject output_dir into the config before running
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["output_dir"] = output_path  # Inject desired output path

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        subprocess.run(
            [sys.executable, "scripts/run_benchmarks.py", "--config", config_path],
            check=True
        )
        return True, 0
    except subprocess.CalledProcessError as e:
        return False, e.returncode
    except Exception as e:
        print(f"❌ Unexpected error on {config_path}: {e}")
        return False, -1

def write_summary(summary, summary_file):
    """Writes a summary CSV file of all benchmark runs."""
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

def write_or_append_summary(row, summary_file, write_header=False):
    """Writes a single row to the summary file, optionally with headers."""
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header and not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ---------------- MAIN ----------------

def run_all_benchmarks(dataset_name):
    config_paths = get_config_paths(dataset_name)
    batch_folder = create_batch_folder(dataset_name)
    summary_file = os.path.join(batch_folder, "benchmark_summary.csv")
    summary = []

    # Copy variation_info.yaml if it exists
    variation_info_src = os.path.join("configs", "generated_configs", dataset_name, "variation_info.yaml")
    variation_info_dst = os.path.join(batch_folder, "variation_info.yaml")
    if os.path.exists(variation_info_src):
        shutil.copy(variation_info_src, variation_info_dst)
        print(f"BATCH : Copied variation_info.yaml to: {variation_info_dst}")
    else:
        print("BATCH : variation_info.yaml not found — batch analysis may fail.")

    print(f"\n BATCH : Starting batch run for {len(config_paths)} configs...")
    print(f" (BATCH)Master output folder: {batch_folder}")

    header_written = False

    for i, config_path in enumerate(config_paths, 1):
        filename = os.path.basename(config_path)
        output_path = os.path.join(batch_folder, filename.replace(".yaml", ""))

        print(f"\n▶ [{i}/{len(config_paths)}] Running: {filename}")
        print(f"BATCH Output: {output_path}")

        os.makedirs(output_path, exist_ok=True)
        success, exit_code = run_config(config_path, output_path)

        row = {
            "config_file": filename,
            "output_path": output_path,
            "success": success,
            "exit_code": exit_code,
            "timestamp": datetime.datetime.now().isoformat()
        }
        write_or_append_summary(row, summary_file, write_header=not header_written)
        header_written = True

    print(f"\n BATCH Summary written to: {summary_file}")

    # Analyze results if possible
    try:
        print("📊 Running batch analysis...")
        #analyze_batch_results(batch_folder)
        analyze_batch_results_from_configs(batch_folder)
        print("✅ Analysis complete.")
    except Exception as e:
        print(f"⚠️ Could not run batch analysis: {e}")

    return batch_folder

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    config_root = "configs/generated_configs"
    datasets = [d for d in os.listdir(config_root) if os.path.isdir(os.path.join(config_root, d))]

    for dataset in datasets:
        print(f"\n Running batch for dataset: {dataset}")
        run_all_benchmarks(dataset)  # Replace if needed
