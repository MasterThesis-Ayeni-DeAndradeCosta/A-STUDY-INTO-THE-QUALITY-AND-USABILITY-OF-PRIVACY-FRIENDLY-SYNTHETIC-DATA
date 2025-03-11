import subprocess
import os
import sys

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")

# Ensure `src/` is in Python path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# def run_anonymization(*args):
#     """
#     Runs the anonymization JAR file using a relative path.
#     """
#     # Construct the relative JAR file path
#     jar_path = os.path.join(project_root, "src", "anonymization", "anonymARXij", "executable", "anonymARXij.jar")

#     # Ensure the JAR file exists
#     if not os.path.exists(jar_path):
#         raise FileNotFoundError(f"JAR file not found: {jar_path}")

#     # Build the command
#     command = ["java", "-jar", jar_path] + list(args)

#     try:
#         # Run the JAR
#         process = subprocess.run(command, capture_output=True, text=True, check=True)

#         # Print outputs
#         print("STDOUT:", process.stdout)
#         print("STDERR:", process.stderr)
#         return process.stdout, process.stderr

#     except subprocess.CalledProcessError as e:
#         print(f"Error running JAR: {e}")
#         return e.stdout, e.stderr
    
# if __name__ == "__main__":
#     # Example usage
#     run_anonymization("arg1", "arg2")

def run_anonymization():
    # Ensure we use absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    jar_path = os.path.abspath(os.path.join(project_root, "src", "anonymization", "anonymARXij", "executable", "anonymARXij.jar"))
    dataset_path = os.path.abspath(os.path.join(project_root, "datasets", "crimeLAPD", "Crime_Data_from_2020_to_Present.csv"))
    config_path = os.path.abspath(os.path.join(project_root, "configs", "config.yaml"))

    # Print the paths for debugging
    print(f"Jar Path: {jar_path}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Config Path: {config_path}")

    # Ensure the JAR file exists before running
    if not os.path.exists(jar_path):
        raise FileNotFoundError(f"JAR file not found: {jar_path}")

    # Build the command
    command = ["java", "-jar", jar_path, dataset_path, config_path]

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == "__main__":
    run_anonymization()