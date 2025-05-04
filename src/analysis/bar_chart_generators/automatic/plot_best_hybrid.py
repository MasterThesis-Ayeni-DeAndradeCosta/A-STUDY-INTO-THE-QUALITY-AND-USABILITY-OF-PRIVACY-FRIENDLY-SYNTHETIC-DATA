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
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\loan_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\loan"


model_to_compare = "RandomForest"

# Create output dir if needed
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, f"original_vs_best_hybrids_{model_to_compare}.png")
table_img_path = os.path.join(output_dir, f"original_vs_best_hybrids_table_{model_to_compare}.png")
table_csv_path = os.path.join(output_dir, f"original_vs_best_hybrids_table_{model_to_compare}.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

# === MAIN ===
df = pd.read_excel(excel_path)
df = df[df["Model"] == model_to_compare]

# Get best original
original_df = df[df["Dataset"] == "Original"]
best_original = original_df.loc[original_df["Average Metric"].idxmax()]
rows = [("Original", best_original)]

# Get best for each HYBRID variant
hybrid_df = df[df["Dataset"].str.contains("_HYBRID", na=False)]
for hybrid_name in hybrid_df["Dataset"].unique():
    variant_df = hybrid_df[hybrid_df["Dataset"] == hybrid_name]
    best_variant = variant_df.loc[variant_df["Average Metric"].idxmax()]
    rows.append((hybrid_name, best_variant))

# === SAVE TABLE ===
combined = pd.DataFrame([r[1] for r in rows])
combined.insert(0, "Source", [r[0] for r in rows])
combined.to_csv(table_csv_path, index=False)
dfi.export(combined, table_img_path)
print(f"📄 Saved table CSV: {table_csv_path}")
print(f"📸 Saved table image: {table_img_path}")

# === PLOT ===
x = np.arange(len(metrics))
bar_width = 0.15

plt.figure(figsize=(10, 5))
for i, (label, row) in enumerate(rows):
    values = [row[m] for m in metrics]
    plt.bar(x + i * bar_width, values, width=bar_width,
            label=label, color=COLOR_MAP.get(label, "#888"), edgecolor="black")
    for j, val in enumerate(values):
        plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

plt.xticks(x + bar_width * (len(rows) - 1) / 2, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"Random Forest Performance on {os.path.basename(excel_path).split('_')[0].capitalize()} Dataset: Original vs Hybrid")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"✅ Saved plot: {plot_path}")
