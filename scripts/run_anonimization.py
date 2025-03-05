import subprocess

jar_path = "C:\Users\isabe\Documents\KUL24_25\MASTER_THESIS\sec_sem\code\thesis_code\thesis_code\A-STUDY-INTO-THE-QUALITY-AND-USABILITY-OF-PRIVACY-FRIENDLY-SYNTHETIC-DATA\src\anonymization\anonymARXij\out\artifacts\anonymARXij.jar"
process = subprocess.run(["java", "-jar", jar_path], capture_output=True, text=True)

print(process.stdout)
