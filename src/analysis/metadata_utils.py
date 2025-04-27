import os

def parse_rows_generated_from_log(log_path):
    """
    Parses the benchmark.log file and returns the true number of rows generated,
    either for pure synthetic models or for hybrid models.
    """
    rows_generated = None

    try:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    # Pure synthetic models (CTGAN, TVAE, GaussianCopula)
                    if "completed:" in line and any(synth in line for synth in ["CTGAN", "TVAE", "GaussianCopula"]):
                        parts = line.strip().split()
                        for i, word in enumerate(parts):
                            if word == "completed:":
                                rows_generated = int(parts[i + 1])
                                break
                    # Hybrid models (CTGAN_HYBRID, TVAE_HYBRID, GaussianCopula_HYBRID)
                    elif "Generated hybrid dataset shape" in line:
                        parts = line.strip().split()
                        for p in parts:
                            if p.startswith("(") and "," in p:
                                rows_generated = int(p.split(",")[0].replace("(", ""))
                                break
    except Exception as e:
        print(f"⚠️ Failed to parse {log_path}: {e}")
        rows_generated = None

    return rows_generated
