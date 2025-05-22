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
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\bar charts\synthetic"
# dataset_name = "loan"

# studentPerformance
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\plots\synthetic"
dataset_name = "StudentPerformance"

# bankMarketing
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\synthetic"
# dataset_name = "bankMarketing"

# censusIncome
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\synthetic"
# dataset_name = "censusIncome"

selected_metric = "MCC"
model_to_compare = "RandomForest"
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
synth_datasets = ["CTGAN", "TVAE", "GaussianCopula"]
top_k = 5
drop_cols = ["suppressed_records", "suppression_percentage", "k_anonymity", "l_diversity", "suppression_limit"]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_excel(excel_path)

# === BAR CHART: BEST OF EACH SYNTH vs ORIGINAL ===
original_df = df[(df["Dataset"] == "Original") & (df["Model"] == model_to_compare)]
best_original = original_df.loc[original_df[selected_metric].idxmax()]
rows = [("Original", best_original)]

for synth in synth_datasets:
    synth_df = df[(df["Dataset"] == synth) & (df["Model"] == model_to_compare)]
    if not synth_df.empty:
        best_synth = synth_df.loc[synth_df[selected_metric].idxmax()]
        rows.append((synth, best_synth))

# === Format combined DataFrame
combined = pd.DataFrame([row[1] for row in rows])
combined.insert(0, "Source", [row[0] for row in rows])
combined.drop(columns=[col for col in drop_cols if col in combined.columns], inplace=True)

# === Export table ===
csv_path = os.path.join(output_dir, f"{dataset_name}_original_vs_best_synthetics_table_{model_to_compare}.csv")
img_path = os.path.join(output_dir, f"{dataset_name}_original_vs_best_synthetics_table_{model_to_compare}.png")
combined.to_csv(csv_path, index=False)
dfi.export(combined.style.set_caption(f"{dataset_name.title()} – Original vs Best Synthetics ({model_to_compare})"), img_path)

# === Plot chart ===
x = np.arange(len(metrics))
bar_width = 0.2
plt.figure(figsize=(10, 5))

for i, row in enumerate(rows):
    label = row[0]
    values = [row[1][metric] for metric in metrics]
    plt.bar(x + i * bar_width, values, width=bar_width,
            label=label, color=COLOR_MAP.get(label, "#999999"), edgecolor="black")
    for j, val in enumerate(values):
        plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha='center', fontsize=8)

plt.xticks(x + bar_width * (len(rows) - 1) / 2, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"{model_to_compare} on {dataset_name.title()} Dataset: Original vs Synthetic")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(rows))
plt.tight_layout(rect=[0, 0.15, 1, 1])

plot_path = os.path.join(output_dir, f"{dataset_name}_original_vs_best_synthetics_{model_to_compare}.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"📊 Saved plot: {plot_path}")

# === TOP-K TABLES PER SYNTHETIC TYPE + COMBINED ===
def export_top_table(synth_name, top_df, suffix):
    top_df_clean = top_df.drop(columns=[c for c in drop_cols if c in top_df.columns])
    csv_path = os.path.join(output_dir, f"{dataset_name}_top{top_k}_{suffix}.csv")
    img_path = os.path.join(output_dir, f"{dataset_name}_top{top_k}_{suffix}.png")
    top_df_clean.to_csv(csv_path, index=False)
    dfi.export(top_df_clean.style.set_caption(f"{dataset_name.title()} Top-{top_k} — {suffix.replace('_', ' ').title()}"), img_path)
    print(f"✅ Saved Top-{top_k} table for: {synth_name}")

top_all = []
for synth in synth_datasets:
    synth_df = df[(df["Dataset"] == synth) & (df["Model"] == model_to_compare)]
    top_k_rows = synth_df.sort_values(selected_metric, ascending=False).head(top_k)
    export_top_table(synth, top_k_rows, suffix=synth.lower())
    top_all.append(top_k_rows)

top_combined = pd.concat(top_all).sort_values(selected_metric, ascending=False).head(top_k)
export_top_table("combined_synthetics", top_combined, suffix="synthetic_all")