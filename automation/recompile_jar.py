import os
import subprocess
import sys

# Resolve paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
java_src = os.path.join(project_root, "src", "anonymization", "anonymARXij")
bin_path = os.path.join(java_src, "bin")
os.makedirs(bin_path, exist_ok=True)

# JAR output path
jar_path = os.path.join(bin_path, "anonymARXij.jar")

# Libraries
lib_dir = os.path.join(project_root, "lib")
arx_jar = os.path.join(lib_dir, "libarx-3.9.1.jar")
yaml_jar = os.path.join(lib_dir, "snakeyaml-2.4.jar")

# OS-dependent separator
sep = ":" if os.name != "nt" else ";"
classpath = sep.join([arx_jar, yaml_jar, bin_path])

# Files to compile
source_files = [
    "src/Main.java",
    "src/AnonymizationManager.java",
    "src/AnonymizationModel.java",
    "src/ConfigLoader.java",
    "src/DataVisualizer.java"
]

def compile_java():
    print("Compiling Java source files...")

    result = subprocess.run(
        ["javac", "-d", "bin", "-sourcepath", "src", "-cp", classpath] + source_files,
        cwd=java_src,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("Compilation failed.")
        print(result.stderr)
        return False
    print("Compilation succeeded.")
    return True

def create_jar():
    print("Creating JAR file...")

    result = subprocess.run(
        ["jar", "cfve", jar_path, "Main", "-C", "bin", "."],
        cwd=java_src,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("JAR creation failed.")
        print(result.stderr)
        return False
    print(f"JAR created at {jar_path}")
    return True

if __name__ == "__main__":
    success = compile_java() and create_jar()
    if success:
        print("Recompile finished successfully.")
    else:
        print("Recompile failed.")
