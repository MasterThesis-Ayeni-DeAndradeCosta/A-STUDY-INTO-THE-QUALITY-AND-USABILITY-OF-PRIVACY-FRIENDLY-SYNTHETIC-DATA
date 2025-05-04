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
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData"


model_to_compare = "RandomForest"

# Create output dir if needed
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, f"original_vs_best_synthetics_{model_to_compare}.png")
table_img_path = os.path.join(output_dir, f"original_vs_best_synthetics_table_{model_to_compare}.png")
table_csv_path = os.path.join(output_dir, f"original_vs_best_synthetics_table_{model_to_compare}.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
synth_datasets = ["CTGAN", "TVAE", "GaussianCopula"]

# === MAIN ===
df = pd.read_excel(excel_path)
original_df = df[(df["Dataset"] == "Original") & (df["Model"] == model_to_compare)]
best_original = original_df.loc[original_df["Average Metric"].idxmax()]

rows = [("Original", best_original)]
for synth in synth_datasets:
    synth_df = df[(df["Dataset"] == synth) & (df["Model"] == model_to_compare)]
    if not synth_df.empty:
        best_synth = synth_df.loc[synth_df["Average Metric"].idxmax()]
        rows.append((synth, best_synth))

# === Format combined dataframe ===
combined = pd.DataFrame([row[1] for row in rows])
combined.insert(0, "Source", [row[0] for row in rows])

# Save as CSV and Image
combined.to_csv(table_csv_path, index=False)
dfi.export(combined, table_img_path)
print(f"📄 Saved table CSV: {table_csv_path}")
print(f"📸 Saved table image: {table_img_path}")

# === PLOT grouped by metric ===
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
plt.title(f"Random Forest Performance on {os.path.basename(excel_path).split('_')[0].capitalize()} Dataset: Original vs Synthetic")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"✅ Saved plot: {plot_path}")
