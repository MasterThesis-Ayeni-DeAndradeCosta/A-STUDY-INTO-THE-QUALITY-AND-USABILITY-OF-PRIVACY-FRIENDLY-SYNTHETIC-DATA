import os
import shutil

def delete_pycache_and_pyc_files():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for root, dirs, files in os.walk(base_path):
        # Remove __pycache__ folders
        for d in dirs:
            if d == "__pycache__":
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"🗑️ Deleted folder: {full_path}")
                except Exception as e:
                    print(f"❌ Failed to delete folder {full_path}. Reason: {e}")

        # Remove .pyc files
        for f in files:
            if f.endswith(".pyc"):
                full_path = os.path.join(root, f)
                try:
                    os.remove(full_path)
                    print(f"🗑️ Deleted file: {full_path}")
                except Exception as e:
                    print(f"❌ Failed to delete file {full_path}. Reason: {e}")

if __name__ == "__main__":
    delete_pycache_and_pyc_files()
