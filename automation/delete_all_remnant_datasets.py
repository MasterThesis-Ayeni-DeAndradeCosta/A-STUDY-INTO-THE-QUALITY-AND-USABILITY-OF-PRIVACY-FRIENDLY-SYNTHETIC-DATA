import os
import shutil

def delete_all_remnant_datasets(base_path="datasets"):
    """
    Deletes all subdirectories in the datasets folder except 'original'.

    Parameters:
    - base_path (str): Path to the datasets directory.
    """
    if not os.path.exists(base_path):
        print(f"⚠️ Datasets folder '{base_path}' not found.")
        return

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item != "original":
            print(f"Deleting folder: {item_path}")
            shutil.rmtree(item_path)

    print("Cleanup complete. Only 'original' remains.")

# Run immediately when script is executed
if __name__ == "__main__":
    delete_all_remnant_datasets()
