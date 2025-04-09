import os

def delete_synthesizers():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthesizers"))
    
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            file_path = os.path.join(base_path, filename)
            if os.path.isfile(file_path) and file_path.endswith(".pkl"):
                try:
                    os.remove(file_path)
                    print(f"Deleted synthesizer: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        print("Synthesizer cleanup complete.")
    else:
        print(f"Folder not found: {base_path}")

if __name__ == "__main__":
    delete_synthesizers()
