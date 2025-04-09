import os

def delete_csv_files_in_folders():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    folders_to_clean = ["anonymized", "cleaned", "synthetic", "train", "test", "hybrid"]

    for folder in folders_to_clean:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path) and file_path.endswith(".csv"):
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            print(f"CSV cleanup done for: {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    delete_csv_files_in_folders()
