import os
import sys

# Ensure import path includes automation
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "automation"))

# Import from sibling cleanup scripts
from delete_remnant_synthesizers import delete_synthesizers
from delete_all_remnant_datasets import delete_csv_files_in_folders
from delete_all_pycache import delete_pycache_and_pyc_files
from delete_remnant_encoders import delete_encoders
<<<<<<< HEAD

=======
from delete_jar_artifacts import delete_jar_files
>>>>>>> dev

def run_cleanup():
    print("Starting full cleanup of artifacts and datasets...\n")
    
    delete_csv_files_in_folders()
    delete_synthesizers()
    delete_pycache_and_pyc_files()
    delete_encoders()
<<<<<<< HEAD
=======
    delete_jar_files()
>>>>>>> dev
    
    print("\n Full cleanup completed.")

if __name__ == "__main__":
    run_cleanup()