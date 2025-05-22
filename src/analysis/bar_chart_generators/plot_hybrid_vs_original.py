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

# loan paths
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\bar charts\hybrid"
# dataset_name = "Loan"

# --- studentPerformance paths (active) ---
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\plots\hybrid"
dataset_name = "studentPerformance"

# bankMarketing paths
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\hybrid"
# dataset_name = "bankMarketing"

# censusIncome
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\plots\hybrid"
# dataset_name = "censusIncome"


# === CONFIG ===
model_to_compare = "RandomForest"  # "LogisticRegression", "RandomForest", …
selected_metric = "MCC"
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
synth_bases = ["CTGAN", "TVAE", "GaussianCopula"]

# Always keep these fields visible
important_config_cols = [
    "k_anonymity", "l_diversity", "suppression_limit", "suppressed_records", "suppression_percentage",
    "test_size", "epochs", "row_multiplier", "rows_generated_at_runtime"
]

os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_excel(excel_path)
df = df[df["Model"] == model_to_compare]

# === PART 1: OVERALL COMPARISON ===
original_df = df[df["Dataset"] == "Original"]
if original_df.empty:
    raise SystemExit("❌ No Original rows found.")
best_original = original_df.loc[original_df[selected_metric].idxmax()]
rows = [("Original", best_original)]

for synth in synth_bases:
    hybrid_label = f"{synth}_HYBRID"
    hybrid_df = df[df["Dataset"] == hybrid_label]
    if hybrid_df.empty:
        print(f"⚠️ No rows for {hybrid_label}, skipping.")
        continue
    best_hybrid = hybrid_df.loc[hybrid_df[selected_metric].idxmax()]
    rows.append((hybrid_label, best_hybrid))

overall_df = pd.DataFrame([r[1] for r in rows])
overall_df.insert(0, "Source", [r[0] for r in rows])

# Export full table
csv_path = os.path.join(output_dir, f"{dataset_name}_bar_chart_rows_{model_to_compare}.csv")
png_path = os.path.join(output_dir, f"{dataset_name}_bar_chart_rows_{model_to_compare}.png")
overall_df.to_csv(csv_path, index=False)
dfi.export(overall_df.style.set_caption(f"{dataset_name.title()} – Hybrid vs Original ({model_to_compare})"), png_path)

# Bar chart
x = np.arange(len(metrics))
bar_width = 0.15
plt.figure(figsize=(10, 5))
for i, (label, row) in enumerate(rows):
    values = [row[m] for m in metrics]
    color = COLOR_MAP.get(label, "#888")
    plt.bar(x + i * bar_width, values, width=bar_width, label=label, color=color, edgecolor="black")
    for j, val in enumerate(values):
        plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
plt.xticks(x + bar_width * (len(rows)-1)/2, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"{model_to_compare} on {dataset_name.title()} Dataset: Original vs Best Hybrids")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(rows))
plt.tight_layout(rect=[0, 0.15, 1, 1])
chart_path = os.path.join(output_dir, f"{dataset_name}_bar_original_vs_hybrids_{model_to_compare}.png")
plt.savefig(chart_path, dpi=300)
plt.close()

# === PART 2: STEPWISE CHAINS ===
for synth in synth_bases:
    step_rows = []

    # Original
    step_rows.append(("Original", best_original))

    # Best HYBRID row
    hybrid_label = synth + "_HYBRID"
    hybrid_df = df[df["Dataset"] == hybrid_label]
    if hybrid_df.empty:
        print(f"⚠️ No {hybrid_label} rows found. Skipping {synth}.")
        continue
    best_hybrid = hybrid_df.loc[hybrid_df[selected_metric].idxmax()]

    # Match Anonymous config to hybrid's k/l/s
    k = best_hybrid.get("k_anonymity")
    l = best_hybrid.get("l_diversity")
    s = best_hybrid.get("suppression_limit")
    anon_df = df[
        (df["Dataset"] == "Anonymous") &
        (df["k_anonymity"] == k) &
        (df["l_diversity"] == l) &
        (df["suppression_limit"] == s)
    ]
    if anon_df.empty:
        print(f"⚠️ No matching Anonymous row for {synth} (k={k}, l={l}, s={s}). Skipping.")
        continue
    best_anon = anon_df.loc[anon_df[selected_metric].idxmax()]

    # Best Synth row
    synth_df = df[df["Dataset"] == synth]
    if synth_df.empty:
        print(f"⚠️ No {synth} rows found. Skipping.")
        continue
    best_synth = synth_df.loc[synth_df[selected_metric].idxmax()]

    # Chain
    step_rows.extend([
        ("Anonymous", best_anon),
        (synth, best_synth),
        (hybrid_label, best_hybrid)
    ])

    # Build table
    full_table = pd.DataFrame([r[1] for r in step_rows])
    full_table.insert(0, "Source", [r[0] for r in step_rows])

    # Force consistent column order
    front_cols = ["Source", "Model", "Dataset"] + metrics + [selected_metric, "AUC-ROC"] + important_config_cols
    full_table = full_table[[col for col in front_cols if col in full_table.columns] + 
                            [col for col in full_table.columns if col not in front_cols]]

    # Export full and clean tables
    base = synth.lower()
    clean_table = full_table.drop(columns=[c for c in full_table.columns if c not in metrics + ["Source", "Model", "Dataset", selected_metric, "AUC-ROC"]])

    clean_csv = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}.csv")
    clean_png = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}.png")
    full_csv = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}_full.csv")
    full_png = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}_full.png")

    clean_table.to_csv(clean_csv, index=False)
    dfi.export(clean_table.style.set_caption(f"{dataset_name.title()} – Stepwise (Metrics Only) – {synth}"), clean_png)
    full_table.to_csv(full_csv, index=False)
    dfi.export(full_table.style.set_caption(f"{dataset_name.title()} – Stepwise (Full Config) – {synth}"), full_png)

    # Stepwise plot
    plt.figure(figsize=(10, 5))
    for i, (label, row) in enumerate(step_rows):
        values = [row[m] for m in metrics]
        color = COLOR_MAP.get(label, "#888")
        plt.bar(x + i * bar_width, values, width=bar_width,
                label=label, color=color, edgecolor="black")
        for j, val in enumerate(values):
            plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    plt.xticks(x + bar_width * (len(step_rows) - 1)/2, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(f"{hybrid_label} – Original → Anonymous → {synth} → Hybrid")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(step_rows))
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    stepwise_chart = os.path.join(output_dir, f"{dataset_name}_stepwise_chain_{base}_chart.png")
    plt.savefig(stepwise_chart, dpi=300)
    plt.close()
