# import subprocess
# import os
# import yaml
# import sys

# # Define correct project paths
# project_root = os.path.abspath(".")  # Root of MasterThesisCode

# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# java_project_path = os.path.join(project_root, "src", "anonymization", "anonymARXij")
# jar_path = os.path.join(java_project_path, "bin", "anonymARXij.jar")
# # Paths to dependencies
# lib_path = os.path.join(project_root, "lib")
# arx_jar = os.path.join(lib_path, "libarx-3.9.1.jar")
# yaml_jar = os.path.join(lib_path, "snakeyaml-2.4.jar")
# # Ensure the bin directory exists for compilation
# bin_path = os.path.join(java_project_path, "bin")
# os.makedirs(bin_path, exist_ok=True)
# # Classpath for Java compilation and execution
# classpath = f"{os.path.abspath(arx_jar)};{os.path.abspath(yaml_jar)};{os.path.abspath(bin_path)}"

import subprocess
import os
import yaml
import sys

# Robustly determine the project root
project_root = os.path.abspath(os.path.dirname(__file__))

# Add project root to sys.path if needed
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Paths for Java project and dependencies
java_project_path = os.path.join(project_root, "src", "anonymization", "anonymARXij")
jar_path = os.path.join(java_project_path, "bin", "anonymARXij.jar")
lib_path = os.path.join(project_root, "lib")
arx_jar = os.path.join(lib_path, "libarx-3.9.1.jar")
yaml_jar = os.path.join(lib_path, "snakeyaml-2.4.jar")
bin_path = os.path.join(java_project_path, "bin")
os.makedirs(bin_path, exist_ok=True)

classpath = f"{os.path.abspath(arx_jar)};{os.path.abspath(yaml_jar)};{os.path.abspath(bin_path)}"


def load_config(config_path="configs/benchmark_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def compile_java():
    """Compiles the Java anonymization code with ARX dependencies."""
    print("\n🔹 Compiling Java project...")
    print("🔹 Using classpath:", classpath)

    compile_process = subprocess.run(
        [
            "javac",
            "-d", "bin",
            "-sourcepath", "src",
            "-cp", classpath,
            "src/Main.java",
            "src/AnonymizationManager.java",
            "src/AnonymizationModel.java",
            "src/ConfigLoader.java",
            "src/DataVisualizer.java"
        ],
        cwd=java_project_path,
        capture_output=True, text=True
    )

    if compile_process.returncode != 0:
        print("❌ Java Compilation Failed!")
        print("STDERR:", compile_process.stderr)
        return False
    return True

def create_jar():
    """Creates a JAR file for the anonymization program."""
    print("\n🔹 Creating JAR file...")

    jar_process = subprocess.run(
        ["jar", "cfve", jar_path, "Main", "-C", "bin", "."],
        cwd=java_project_path,
        capture_output=True, text=True
    )

    if jar_process.returncode != 0:
        print("❌ JAR Creation Failed!")
        print("STDERR:", jar_process.stderr)
        return False
    return True

# def run_anonymization():
#     """Runs the Java anonymization JAR."""

#     if not compile_java():
#         print("⛔ Skipping JAR creation due to compilation errors.")
#         return False
#     if not create_jar():
#         print("⛔ Skipping execution due to JAR creation errors.")
#         return False

#     print("\n🔹 Running Java Anonymization Program...")
#     run_process = subprocess.run(
#         ["java", "-cp", f"{jar_path};{classpath}", "Main"],  # ✅ No dataset/output path needed
#         capture_output=True, text=True
#     )

#     print("\n✅ STDOUT:", run_process.stdout)
#     print("❗ STDERR:", run_process.stderr)

#     return run_process.returncode == 0

def run_anonymization(dataset_path=None, config_path="configs/benchmark_config.yaml"):
    config = load_config(config_path)
    enable_anonymization = config["anonymization"].get("enable_anonymization", False)
    if not enable_anonymization:
        print("\nAnonymization Skipped (Disabled in Configuration).")
        return True  # Consider True for "skipped but not failed"
    """Runs the Java anonymization JAR using the cleaned dataset path."""
    if not compile_java():
        print("⛔ Skipping JAR creation due to compilation errors.")
        return False
    if not create_jar():
        print("⛔ Skipping execution due to JAR creation errors.")
        return False
    print("\n🔹 Running Java Anonymization Program...")
    # Fallback to original dataset if none provided
    if dataset_path is None:
        config = load_config()
        dataset_path = os.path.abspath(config["dataset"]["path"])
    run_process = subprocess.run(
        ["java", "-cp", f"{jar_path};{classpath}", "Main", dataset_path],  # Pass dataset path
        capture_output=True, text=True
    )
    print("\n✅ STDOUT:", run_process.stdout)
    print("❗ STDERR:", run_process.stderr)

    return run_process.returncode == 0

# Example usage
if __name__ == "__main__":
    success = run_anonymization()
    if success:
        print("Anonymization completed successfully!")
    else:
        print("Anonymization failed!")
