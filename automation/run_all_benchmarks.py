import os
import subprocess
import datetime

# ---------------- SETTINGS ----------------

def get_config_paths(dataset_name):
    config_dir = f"configs/generated_configs/{dataset_name}"
    config_files = [
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith(".yaml")
    ]
    return config_files

def create_batch_folder(dataset_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join("outputs", "batch", f"{dataset_name}_batch_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder

def run_config(config_path, output_path):
    subprocess.run([
        "python", "scripts/run_benchmarks.py",
        "--config", config_path,
        "--output", output_path
    ])

# ---------------- MAIN ----------------

def run_all_benchmarks(dataset_name):
    config_paths = get_config_paths(dataset_name)
    batch_folder = create_batch_folder(dataset_name)

    print(f"\n🚀 Starting batch run for {len(config_paths)} configs...")
    print(f"📁 Master output folder: {batch_folder}")

    for i, config_path in enumerate(config_paths, 1):
        filename = os.path.basename(config_path)
        output_path = os.path.join(batch_folder, filename.replace(".yaml", ""))
        
        print(f"\n▶ [{i}/{len(config_paths)}] Running: {filename}")
        print(f"📂 Output: {output_path}")
        
        run_config(config_path, output_path)

    print(f"\n✅ All runs complete. Results saved to: {batch_folder}")
    return batch_folder

# ---------------- ENTRY ----------------

if __name__ == "__main__":
    run_all_benchmarks(dataset_name="loan")  # You can replace this with any dataset name
