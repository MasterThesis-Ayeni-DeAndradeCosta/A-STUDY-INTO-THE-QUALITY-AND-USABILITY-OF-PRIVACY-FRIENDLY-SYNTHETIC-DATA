# delete_remnant_encoders.py
import os

def delete_encoders():
    """
    Deletes all encoder .pkl files saved in the artifacts/ directory.
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            if os.path.isfile(os.path.join(base_path, filename)) and filename.endswith("_encoder.pkl"):
                file_path = os.path.join(base_path, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted encoder: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}. Reason: {e}")
        print("Encoder cleanup complete.")
    else:
        print(f"Folder not found: {base_path}")

if __name__ == "__main__":
    delete_encoders()