import os
import re
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional

# Regex that matches e.g. "2025-04-11 14:57:20,005"
TS_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")

def parse_timestamp(line: str) -> Optional[datetime]:
    """Extract a datetime from the front of a log line or None if not matched."""
    match = TS_REGEX.match(line)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return None

def humanize_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h} hours {m} minutes {s} secs"

def extract_runtime_from_log(log_path: str) -> Dict[str, Any]:
    """
    Reads one benchmark.log, finds relevant timestamps for:
      - Preprocessing
      - Anonymization
      - Synthesis total
      - CTGAN fit→sample
      - TVAE fit→sample
      - GaussianCopula fit→sample
      - ML training
      - ML evaluation
      - total time
    Returns them as numeric durations (seconds) or None if not found.
    """
    # We'll store raw timestamps as optional datetimes
    t: Dict[str, Optional[datetime]] = {
        "start": None,               # "Benchmarking started"
        "preprocess_done": None,     # "Preprocessing completed."
        "anon_start": None,          # "Running Java Anonymization Program"
        "anon_end": None,            # "Java anonymization completed"
        "synth_start": None,         # "Starting Synthetic Data Generation Pipeline"
        "synth_end": None,           # "Synthetic Data Generation Completed."
        "ctgan_fit": None,           # "'EVENT': 'Fit' ... CTGANSynthesizer"
        "ctgan_sample": None,        # "'EVENT': 'Sample' ... CTGANSynthesizer"
        "tvae_fit": None,            # "'EVENT': 'Fit' ... TVAESynthesizer"
        "tvae_sample": None,         # "'EVENT': 'Sample' ... TVAESynthesizer"
        "gau_fit": None,             # "'EVENT': 'Fit' ... GaussianCopulaSynthesizer"
        "gau_sample": None,          # "'EVENT': 'Sample' ... GaussianCopulaSynthesizer"
        "train_start": None,         # "Training models on all datasets"
        "train_end": None,           # "Model Training Completed."
        "eval_start": None,          # "Evaluating models..."
        "eval_end": None,            # "Model Training and Evaluation completed"
        "end": None,                 # "Benchmarking Completed."
    }

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            dt = parse_timestamp(line)
            if not dt:
                continue

            # Overall start/end
            if "Benchmarking started for dataset:" in line:
                t["start"] = dt
            elif "Benchmarking Completed." in line:
                t["end"] = dt

            # Preprocessing
            elif "Preprocessing completed." in line:
                t["preprocess_done"] = dt

            # Anonymization
            elif "Running Java Anonymization Program" in line:
                t["anon_start"] = dt
            elif "Java anonymization completed" in line:
                t["anon_end"] = dt

            # Synthesis
            elif "Starting Synthetic Data Generation Pipeline" in line:
                t["synth_start"] = dt
            elif "Synthetic Data Generation Completed." in line:
                t["synth_end"] = dt

            # Synthesizer fit/sample
            elif "'EVENT': 'Fit'" in line and "CTGANSynthesizer" in line:
                t["ctgan_fit"] = dt
            elif "'EVENT': 'Sample'" in line and "CTGANSynthesizer" in line:
                t["ctgan_sample"] = dt

            elif "'EVENT': 'Fit'" in line and "TVAESynthesizer" in line:
                t["tvae_fit"] = dt
            elif "'EVENT': 'Sample'" in line and "TVAESynthesizer" in line:
                t["tvae_sample"] = dt

            elif "'EVENT': 'Fit'" in line and "GaussianCopulaSynthesizer" in line:
                t["gau_fit"] = dt
            elif "'EVENT': 'Sample'" in line and "GaussianCopulaSynthesizer" in line:
                t["gau_sample"] = dt

            # ML training/eval
            elif "Training models on all datasets" in line:
                t["train_start"] = dt
            elif "Model Training Completed." in line:
                t["train_end"] = dt
            elif "Evaluating models..." in line:
                t["eval_start"] = dt
            elif "Model Training and Evaluation completed" in line:
                t["eval_end"] = dt

    def dur(k1: str, k2: str) -> Optional[float]:
        """
        Return the duration in seconds from t[k1] to t[k2], or None if either is missing.
        """
        start_dt = t[k1]
        end_dt   = t[k2]
        if start_dt is not None and end_dt is not None:
            return round((end_dt - start_dt).total_seconds(), 2)
        return None

    # Now build final numeric durations
    result: Dict[str, Any] = {}
    result["total_sec"]          = dur("start", "end")
    result["preprocessing_sec"]  = dur("start", "preprocess_done")
    result["anonymization_sec"]  = dur("anon_start", "anon_end")
    result["synthesis_sec"]      = dur("synth_start", "synth_end")
    # per-synth
    result["ctgan_sec"]          = dur("ctgan_fit", "ctgan_sample")
    result["tvae_sec"]           = dur("tvae_fit", "tvae_sample")
    result["gaussian_sec"]       = dur("gau_fit", "gau_sample")
    # ML train/eval
    result["train_sec"]          = dur("train_start", "train_end")
    result["evaluation_sec"]     = dur("eval_start", "eval_end")

    return result

def collect_runtimes_for_batch(batch_dir: str) -> pd.DataFrame:
    """
    For each subfolder in 'batch_dir', parse its benchmark.log to get durations.
    Return a DataFrame of results (one row per subfolder).
    """
    rows = []
    for run_name in os.listdir(batch_dir):
        run_path = os.path.join(batch_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        log_path = os.path.join(run_path, "benchmark.log")
        if not os.path.isfile(log_path):
            continue

        # Extract run times
        row = extract_runtime_from_log(log_path)
        # Store the run_name so we can see which subfolder it is
        row["run_name"] = run_name
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Reorder columns
    col_order = [
        "run_name",
        "total_sec",
        "preprocessing_sec",
        "anonymization_sec",
        "synthesis_sec",
        "ctgan_sec",
        "tvae_sec",
        "gaussian_sec",
        "train_sec",
        "evaluation_sec"
    ]
    # Keep only the columns that exist
    final_cols = [c for c in col_order if c in df.columns]
    return df[final_cols]

def main():
    parser = argparse.ArgumentParser(description="Collect runtime durations across a batch of runs.")
    parser.add_argument("--batch_dir", required=True, help="Path to the batch output folder.")
    args = parser.parse_args()

    batch_dir = os.path.abspath(args.batch_dir)
    if not os.path.isdir(batch_dir):
        print(f"❌ Not a valid directory: {batch_dir}")
        sys.exit(1)

    df = collect_runtimes_for_batch(batch_dir)
    if df.empty:
        print("⚠️ No runs with logs found in that batch.")
        return

    # We'll sum up each numeric column to produce a “BatchTotals” row
    numeric_cols = [
        "total_sec","preprocessing_sec","anonymization_sec",
        "synthesis_sec","ctgan_sec","tvae_sec","gaussian_sec",
        "train_sec","evaluation_sec"
    ]
    sums = df[numeric_cols].sum(numeric_only=True)

    # Build a new row
    totals_row: Dict[str, Any] = {"run_name": "BatchTotals"}
    for c in numeric_cols:
        val = sums.get(c, 0.0)
        totals_row[c] = None if pd.isna(val) else round(float(val), 2)

    # Append that row with pd.concat (since .append is removed)
    total_df = pd.DataFrame([totals_row])
    df = pd.concat([df, total_df], ignore_index=True)

    # Add human-readable version under each numeric column
    readable_row: Dict[str, Any] = {"run_name": ""}
    for c in numeric_cols:
        val = totals_row.get(c)
        readable_row[c] = humanize_seconds(val) if val is not None else ""
    df = pd.concat([df, pd.DataFrame([readable_row])], ignore_index=True)

    # Write to CSV
    dataset_name = os.path.basename(batch_dir).replace("batch_", "")
    filename = f"{dataset_name}_runtime_summary.csv"
    analysis_dir = os.path.join(batch_dir, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    outpath = os.path.join(analysis_dir, filename)
    df.to_csv(outpath, index=False)
    print(f"✅ Created {outpath}")

if __name__ == "__main__":
    main()
