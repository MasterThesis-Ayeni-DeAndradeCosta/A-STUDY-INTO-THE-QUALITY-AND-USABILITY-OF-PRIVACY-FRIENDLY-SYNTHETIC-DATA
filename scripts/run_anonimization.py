import subprocess
import os

# Paths
java_project_path = r"C:\Users\isabe\Documents\KUL24_25\MASTER_THESIS\sec_sem\code\thesis_code\thesis_code\A-STUDY-INTO-THE-QUALITY-AND-USABILITY-OF-PRIVACY-FRIENDLY-SYNTHETIC-DATA\src\anonymization\anonymARXij"
jar_path = os.path.join(java_project_path, "src", "executable", "anonymARXij.jar")

# Paths to dependencies (Now in `anonymARXij/libraries`)
lib_path = os.path.join(java_project_path, "libraries")
arx_jar = os.path.join(lib_path, "arx-3.9.1.jar")
yaml_jar = os.path.join(lib_path, "snakeyaml-2.4.jar")

# Ensure that libraries folder is included in the classpath
classpath = f"{lib_path}/*;."

# Compile Java Project
print("Compiling Java project...")
compile_process = subprocess.run(
    ["javac", "-d", "bin", "-sourcepath", "src", "-cp", classpath, "src/Main.java"],
    cwd=java_project_path,
    capture_output=True, text=True
)

if compile_process.returncode != 0:
    print("Java Compilation Failed!")
    print("STDERR:", compile_process.stderr)
    exit(1)

# Create JAR File
print("Creating JAR file...")
jar_process = subprocess.run(
    ["jar", "cfve", jar_path, "Main", "-C", "bin", "."],
    cwd=java_project_path,
    capture_output=True, text=True
)

if jar_process.returncode != 0:
    print("JAR Creation Failed!")
    print("STDERR:", jar_process.stderr)
    exit(1)

# Run the JAR
print("Running Java Program...")
run_process = subprocess.run(
    ["java", "-cp", f"{jar_path};{classpath}", "Main"],
    capture_output=True, text=True, cwd=java_project_path  # Run from `anonymARXij/`
)



print("STDOUT:", run_process.stdout)
print("STDERR:", run_process.stderr)
