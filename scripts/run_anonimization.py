import subprocess
import os

jar_path = r"C:\Users\isabe\Documents\KUL24_25\MASTER_THESIS\sec_sem\code\thesis_code\thesis_code\A-STUDY-INTO-THE-QUALITY-AND-USABILITY-OF-PRIVACY-FRIENDLY-SYNTHETIC-DATA\src\anonymization\anonymARXij\src\executable\anonymARXij.jar"
working_directory = r"C:\Users\isabe\Documents\KUL24_25\MASTER_THESIS\sec_sem\code\thesis_code\thesis_code\A-STUDY-INTO-THE-QUALITY-AND-USABILITY-OF-PRIVACY-FRIENDLY-SYNTHETIC-DATA\src\anonymization\anonymARXij"  # Change this if needed

process = subprocess.run(["java", "-jar", jar_path], capture_output=True, text=True, cwd=working_directory)

print("STDOUT:", process.stdout)
print("STDERR:", process.stderr)
