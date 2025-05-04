import os
import re
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional

TS_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
#DURATION_REGEX = re.compile(r"Fitting complete for (CTGAN|TVAE|GaussianCopula)\. Duration: ([\d.]+) seconds\.")
DURATION_REGEX = re.compile(r"(?:\[SYNTHETIC\] )?Fitting complete for (CTGAN|TVAE|GaussianCopula)\. Duration: ([\d.]+) seconds\.")

HYBRID_DURATION_REGEX = re.compile(r"Hybrid pipeline duration: ([\d.]+) seconds\.")
EVALUATION_DURATION_REGEX = re.compile(r"Model evaluation completed in ([\d.]+) seconds\.")

def parse_timestamp(line: str) -> Optional[datetime]:
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
    t: Dict[str, Optional[datetime]] = {
        "benchmark_start": None,
        "benchmark_end": None,
        "preprocess_start": None,
        "preprocess_done": None,
        "anon_start": None,
        "anon_end": None,
        "synth_start": None,
        "synth_end": None,
        "hybrid_start": None,
        "hybrid_end": None,
        "train_start": None,
        "train_end": None,
    }
    durations: Dict[str, Optional[float]] = {
        "ctgan_sec": None,
        "tvae_sec": None,
        "gaussian_sec": None,
        "hybrid_ctgan_sec": None,
        "hybrid_tvae_sec": None,
        "hybrid_gaussian_sec": None,
        "hybrid_duration_sec": None,
        "evaluation_sec": None
    }

    in_hybrid_block = False

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            dt = parse_timestamp(line)
            if not dt:
                continue

            if "Benchmarking started for dataset:" in line:
                t["benchmark_start"] = dt
            elif "Benchmarking Completed." in line:
                t["benchmark_end"] = dt

            if "Running preprocessing..." in line:
                t["preprocess_start"] = dt
            elif "Preprocessing completed." in line:
                t["preprocess_done"] = dt

            elif "Starting Anonymization Pipeline" in line:
                t["anon_start"] = dt
            elif "Postprocessing report saved at" in line:
                t["anon_end"] = dt

            elif "Starting standard Synthetic Data Generation pipeline..." in line:
                t["synth_start"] = dt
            elif "Synthetic data generated with" in line:
                t["synth_end"] = dt

            elif "[HYBRID] Starting synthetic generation for hybrid pipeline..." in line:
                t["hybrid_start"] = dt
                in_hybrid_block = True
            elif "Hybrid pipeline completed successfully." in line:
                t["hybrid_end"] = dt
                in_hybrid_block = False

            elif "[TRAINING] Training models from scratch..." in line:
                t["train_start"] = dt
            elif "[UTILITY] Model Training Completed." in line:
                t["train_end"] = dt

            m = DURATION_REGEX.search(line)
            if m:
                model, secs = m.group(1), float(m.group(2))
                model = model.lower()
                if in_hybrid_block:
                #if "[HYBRID]" in line:
                    durations[f"hybrid_{model}_sec"] = secs
                else:
                    durations[f"{model}_sec"] = secs

            m_hybrid = HYBRID_DURATION_REGEX.search(line)
            if m_hybrid:
                durations["hybrid_duration_sec"] = float(m_hybrid.group(1))

            m_eval = EVALUATION_DURATION_REGEX.search(line)
            if m_eval:
                durations["evaluation_sec"] = float(m_eval.group(1))

    def dur(k1: str, k2: str) -> Optional[float]:
        start_dt = t[k1]
        end_dt = t[k2]
        if start_dt is not None and end_dt is not None:
            return round((end_dt - start_dt).total_seconds(), 2)
        return None

    result: Dict[str, Any] = {}
    result["runtime_sec"] = dur("benchmark_start", "benchmark_end")
    result["preprocessing_sec"] = dur("preprocess_start", "preprocess_done")
    result["anonymization_sec"] = dur("anon_start", "anon_end")
    result["synthesis_sec"] = dur("synth_start", "synth_end")
    result["hybrid_synthesis_sec"] = dur("hybrid_start", "hybrid_end")
    result.update(durations)
    result["model_training_sec"] = dur("train_start", "train_end")

    return result

def collect_runtimes_for_batch(batch_dir: str) -> pd.DataFrame:
    rows = []
    for run_name in os.listdir(batch_dir):
        run_path = os.path.join(batch_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        log_path = os.path.join(run_path, "benchmark.log")
        if not os.path.isfile(log_path):
            continue

        row = extract_runtime_from_log(log_path)
        row["run_name"] = run_name
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    time_cols = [
        "preprocessing_sec", "anonymization_sec", "synthesis_sec", "ctgan_sec", "tvae_sec", "gaussian_sec",
        "hybrid_synthesis_sec", "hybrid_ctgan_sec", "hybrid_tvae_sec", "hybrid_gaussian_sec", "model_training_sec", "evaluation_sec"
    ]
    df["total_runtime_sec"] = df[time_cols].sum(axis=1, skipna=True)
    df["total_runtime_human"] = df["total_runtime_sec"].apply(lambda x: humanize_seconds(x) if pd.notnull(x) else "")

    numeric_cols = [col for col in df.columns if col.endswith("_sec")]
    sums = df[numeric_cols].sum(numeric_only=True)
    totals_row: Dict[str, Any] = {"run_name": "BatchTotals"}
    for c in numeric_cols:
        val = sums.get(c, 0.0)
        totals_row[c] = None if pd.isna(val) else round(float(val), 2)

    totals_row["total_runtime_human"] = humanize_seconds(totals_row["total_runtime_sec"]) if totals_row.get("total_runtime_sec") else ""

    df = pd.concat([df, pd.DataFrame([totals_row])], ignore_index=True)

     # 🆕 Add the human-readable row below totals
    readable_row: Dict[str, Any] = {"run_name": ""}
    for c in numeric_cols:
        val = totals_row.get(c)
        readable_row[c] = humanize_seconds(val) if val is not None else ""
    readable_row["total_runtime_human"] = totals_row.get("total_runtime_human", "")
    
    df = pd.concat([df, pd.DataFrame([readable_row])], ignore_index=True)


    col_order = [
        "run_name",
        "runtime_sec",
        "preprocessing_sec",
        "anonymization_sec",
        "synthesis_sec",
        "ctgan_sec",
        "tvae_sec",
        "gaussian_sec",
        "hybrid_synthesis_sec",
        "hybrid_duration_sec",
        "hybrid_ctgan_sec",
        "hybrid_tvae_sec",
        "hybrid_gaussian_sec",
        "model_training_sec",
        "evaluation_sec",
        "total_runtime_sec",
        "total_runtime_human"
    ]
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

    dataset_name = os.path.basename(batch_dir).replace("batch_", "")
    analysis_dir = os.path.join(batch_dir, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    outpath = os.path.join(analysis_dir, f"{dataset_name}_runtime_summary.xlsx")
    df.to_excel(outpath, index=False)

    print(f"✅ Created {outpath}")

if __name__ == "__main__":
    main()
