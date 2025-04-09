import os

def delete_csv_files():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    folders_to_clean = ["anonymized", "cleaned", "synthetic", "train", "test", "hybrid"]
    for folder in folders_to_clean:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    os.remove(os.path.join(folder_path, filename))
                    print(f"Deleted CSV: {filename}")

def delete_synthesizer_files():
    synth_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthesizers"))
    if os.path.exists(synth_path):
        for filename in os.listdir(synth_path):
            if filename.endswith(".pkl"):
                os.remove(os.path.join(synth_path, filename))
                print(f"Deleted Synthesizer: {filename}")

def run_cleanup():
    print("Starting cleanup of all remnants...")
    delete_csv_files()
    delete_synthesizer_files()
    print("Cleanup completed.")

if __name__ == "__main__":
    run_cleanup()
