# ✅ plot_best_anonymized.py
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

plot_path = os.path.join(output_dir, f"original_vs_best_anonymized_{model_to_compare}.png")
table_img_path = os.path.join(output_dir, f"original_vs_best_anonymized_table_{model_to_compare}.png")
table_csv_path = os.path.join(output_dir, f"original_vs_best_anonymized_table_{model_to_compare}.csv")
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

# === MAIN ===
df = pd.read_excel(excel_path)
original_df = df[(df["Dataset"] == "Original") & (df["Model"] == model_to_compare)]
anon_df = df[(df["Dataset"] == "Anonymous") & (df["Model"] == model_to_compare)]

best_original = original_df.loc[original_df["Average Metric"].idxmax()]
best_anon = anon_df.loc[anon_df["Average Metric"].idxmax()]

combined = pd.concat([
    best_original.to_frame().T.assign(Source="Original"),
    best_anon.to_frame().T.assign(Source="Anonymous")
])
combined.reset_index(drop=True, inplace=True)

# === SAVE TABLE CSV AND IMAGE ===
combined.to_csv(table_csv_path, index=False)
dfi.export(combined, table_img_path)
print(f"📄 Saved table CSV: {table_csv_path}")
print(f"📸 Saved table image: {table_img_path}")

# === PLOT ===
x = np.arange(len(metrics))
bar_width = 0.35
labels = combined["Source"]
data = [combined.loc[i, metrics].values for i in combined.index]

plt.figure(figsize=(10, 5))
for i in range(len(data)):
    plt.bar(x + i * bar_width, data[i], width=bar_width,
            color=COLOR_MAP.get(labels[i], "#CCCCCC"),
            label=labels[i], edgecolor="black")
    for j, val in enumerate(data[i]):
        plt.text(x[j] + i * bar_width, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

plt.xticks(x + bar_width / 2, metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"Best Original vs Anonymized ({model_to_compare})")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"✅ Saved plot: {plot_path}")
