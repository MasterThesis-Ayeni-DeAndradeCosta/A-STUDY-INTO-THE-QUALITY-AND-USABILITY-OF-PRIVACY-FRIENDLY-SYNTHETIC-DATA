import os
import subprocess
import datetime
import csv
import yaml
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed #for parallel execution
from multiprocessing import cpu_count
import signal
import psutil
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
        if f.endswith(".yaml") and f != "variation_info.yaml"
    ]

def create_batch_folder(dataset_name):
    """Creates and returns a timestamped output folder for the batch run."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join("outputs", "batch", f"{dataset_name}_batch_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder

def run_config(config_path, output_path):
    """Runs a single benchmark configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["output_dir"] = output_path

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
        print(f"[BATCH] Unexpected error on {config_path}: {e}")
        return False, -1


def write_summary(summary, summary_file):
    """Writes a summary CSV file of all benchmark runs, including success stats at the end."""
    fieldnames = summary[0].keys()
    success_count = sum(1 for row in summary if row["success"])
    fail_count = len(summary) - success_count

    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

        # Write a separator row and summary counts
        writer.writerow({})  # empty row for readability
        writer.writerow({"config_file": "TOTAL_RUNS", "output_path": len(summary)})
        writer.writerow({"config_file": "SUCCEEDED", "output_path": success_count})
        writer.writerow({"config_file": "FAILED", "output_path": fail_count})


# ---------------- MAIN ----------------

def run_benchmark_batch_parallel_execution(dataset_name, max_workers=None):
    config_paths = get_config_paths(dataset_name)
    batch_folder = create_batch_folder(dataset_name)
    summary_file = os.path.join(batch_folder, "benchmark_summary.csv")
    summary = []

    variation_info_src = os.path.join("configs", "generated_configs", dataset_name, "variation_info.yaml")
    variation_info_dst = os.path.join(batch_folder, "variation_info.yaml")
    if os.path.exists(variation_info_src):
        shutil.copy(variation_info_src, variation_info_dst)
        print(f"BATCH Copied variation_info.yaml to: {variation_info_dst}")
    else:
        print("BATCH variation_info.yaml not found — batch analysis may fail.")

    print(f"\n [BATCH] Starting batch run for {len(config_paths)} configs...")

    max_workers = max_workers or max(1, os.cpu_count() - 2)
    print(f" cpu count is {os.cpu_count()}, using max_workers={max_workers}")

    future_to_meta = {}

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, config_path in enumerate(config_paths, 1):
                filename = os.path.basename(config_path)
                output_path = os.path.join(batch_folder, filename.replace(".yaml", ""))

                print(f"\n▶ [{i}/{len(config_paths)}] Running: {filename}")
                print(f" BATCH Output: {output_path}")

                os.makedirs(output_path, exist_ok=True)

                future = executor.submit(run_config, config_path, output_path)
                future_to_meta[future] = {
                    "config_file": filename,
                    "output_path": output_path
                }

            for future in as_completed(future_to_meta):
                meta = future_to_meta[future]
                try:
                    success, exit_code = future.result()
                except Exception as e:
                    success, exit_code = False, -1
                    print(f" Exception in {meta['config_file']}: {e}")

                meta.update({
                    "success": success,
                    "exit_code": exit_code,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                summary.append(meta)

    except KeyboardInterrupt:
        print("\n Batch runner interrupted by user. Terminating subprocesses...")

        # Kill all subprocesses cleanly
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass

    finally:
        print(" Writing BATCH summary...")
        if summary:
            write_summary(summary, summary_file)
            print(f" BATCH Summary written to: {summary_file}")
        else:
            print(" No BATCH runs completed. No summary written.")

    return batch_folder

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    config_root = "configs/generated_configs"
    datasets = [d for d in os.listdir(config_root) if os.path.isdir(os.path.join(config_root, d))]

    for dataset in datasets:
        print(f"\n Running batch for dataset: {dataset}")
        run_benchmark_batch_parallel_execution(dataset)
