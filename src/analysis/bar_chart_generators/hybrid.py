import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import dataframe_image as dfi

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from color_palette import COLOR_MAP

# === CONFIG ===
dataset_name = "censusIncome"
model_to_compare = "LogisticRegression"
ranking_metric = "F1"  # or MCC, etc.

excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\plots\hybrid"

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
synth_bases = ["CTGAN", "TVAE", "GaussianCopula"]
important_config_cols = [
    "k_anonymity", "l_diversity", "suppression_limit", "suppressed_records", "suppression_percentage",
    "test_size", "epochs", "row_multiplier", "rows_generated_at_runtime"
]

os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_excel(excel_path)
df = df[df["Model"] == model_to_compare]

# === GET BEST ORIGINAL ===
original_df = df[df["Dataset"] == "Original"]
if original_df.empty or original_df[ranking_metric].dropna().empty:
    raise SystemExit(f"❌ No valid Original rows found for metric '{ranking_metric}'.")
best_original = original_df.loc[original_df[ranking_metric].idxmax()]

# === OVERALL COMPARISON TABLE ===
rows = [("Original", best_original)]
for synth in synth_bases:
    hybrid_label = f"{synth}_HYBRID"
    hybrid_df = df[df["Dataset"] == hybrid_label]
    if hybrid_df.empty or hybrid_df[ranking_metric].dropna().empty:
        print(f"⚠️ No valid {hybrid_label} rows found.")
        continue
    best_hybrid = hybrid_df.loc[hybrid_df[ranking_metric].idxmax()]
    rows.append((hybrid_label, best_hybrid))

overall_df = pd.DataFrame([r[1] for r in rows])
overall_df.insert(0, "Source", [r[0] for r in rows])
summary_csv = os.path.join(output_dir, f"{dataset_name}_best_hybrids_vs_original.csv")
summary_png = os.path.join(output_dir, f"{dataset_name}_best_hybrids_vs_original.png")
overall_df.to_csv(summary_csv, index=False)
dfi.export(overall_df.style.set_caption(f"{dataset_name.title()} – Original vs Best Hybrids"), summary_png)

# === STEPWISE CHAINS ===
for synth in synth_bases:
    step_rows = [("Original", best_original)]

    hybrid_label = synth + "_HYBRID"
    hybrid_df = df[df["Dataset"] == hybrid_label]
    if hybrid_df.empty or hybrid_df[ranking_metric].dropna().empty:
        print(f"⚠️ No {hybrid_label} rows found or no valid '{ranking_metric}' values. Skipping hybrid step for {synth}.")
        best_hybrid = None
    else:
        best_hybrid = hybrid_df.loc[hybrid_df[ranking_metric].idxmax()]

    if best_hybrid is not None:
        k = best_hybrid.get("k_anonymity")
        l = best_hybrid.get("l_diversity")
        s = best_hybrid.get("suppression_limit")
        anon_df = df[
            (df["Dataset"] == "Anonymous") &
            (df["k_anonymity"] == k) &
            (df["l_diversity"] == l) &
            (df["suppression_limit"] == s)
        ]
        if anon_df.empty or anon_df[ranking_metric].dropna().empty:
            print(f"⚠️ No matching Anonymous row for {synth} (k={k}, l={l}, s={s}). Using best available Anonymous.")
            anon_df = df[df["Dataset"] == "Anonymous"]
            best_anon = anon_df.loc[anon_df[ranking_metric].idxmax()] if not anon_df.empty else None
        else:
            best_anon = anon_df.loc[anon_df[ranking_metric].idxmax()]
    else:
        best_anon = None

    if best_anon is not None:
        step_rows.append(("Anonymous", best_anon))
    if best_hybrid is not None:
        step_rows.append((hybrid_label, best_hybrid))

    if len(step_rows) < 2:
        print(f"⚠️ Not enough valid data for {synth} to generate stepwise chain.")
        continue

    # === EXPORT TABLES ===
    full_table = pd.DataFrame([r[1] for r in step_rows])
    full_table.insert(0, "Source", [r[0] for r in step_rows])
    front_cols = ["Source", "Model", "Dataset"] + metrics + ["Average Metric", "AUC-ROC"] + important_config_cols
    full_table = full_table[[c for c in front_cols if c in full_table.columns] + 
                            [c for c in full_table.columns if c not in front_cols]]

    base = synth.lower()
    clean_table = full_table.drop(columns=[c for c in full_table.columns if c not in metrics + ["Source", "Model", "Dataset", "Average Metric", "AUC-ROC"]])
    clean_csv = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}.csv")
    clean_png = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}.png")
    full_csv = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}_full.csv")
    full_png = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}_full.png")
    clean_table.to_csv(clean_csv, index=False)
    dfi.export(clean_table.style.set_caption(f"{dataset_name.title()} – Stepwise (Metrics Only) – {synth}"), clean_png)
    full_table.to_csv(full_csv, index=False)
    dfi.export(full_table.style.set_caption(f"{dataset_name.title()} – Stepwise (Full Config) – {synth}"), full_png)

    # === PLOT ===
    x = np.arange(len(metrics))
    bar_width = 0.2
    plt.figure(figsize=(10, 5))
    for i, (label, row) in enumerate(step_rows):
        values = [row[m] for m in metrics]
        color = COLOR_MAP.get(label, "#888")
        plt.bar(x + i * bar_width, values, width=bar_width, label=label, color=color, edgecolor="black")
        for j, val in enumerate(values):
            plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    plt.xticks(x + bar_width * (len(step_rows) - 1) / 2, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{hybrid_label} – Original → Anonymous → Hybrid")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(step_rows))
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    stepwise_chart = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}_chart.png")
    plt.savefig(stepwise_chart, dpi=300)
    plt.close()
