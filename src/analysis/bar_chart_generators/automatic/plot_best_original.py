# ✅ plot_best_original.py
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
model_to_compare = "RandomForest"
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData"



# Create output dir if needed
os.makedirs(output_dir, exist_ok=True)

plot_path = os.path.join(output_dir, f"best_original_{model_to_compare}.png")
table_img_path = os.path.join(output_dir, f"best_original_table_{model_to_compare}.png")
table_csv_path = os.path.join(output_dir, f"best_original_table_{model_to_compare}.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

# === MAIN ===
df = pd.read_excel(excel_path)
original_df = df[(df["Dataset"] == "Original") & (df["Model"] == model_to_compare)]
best_row = original_df.loc[original_df["Average Metric"].idxmax()]
combined = best_row.to_frame().T.assign(Source="Original")

# === SAVE TABLE CSV AND IMAGE ===
combined.to_csv(table_csv_path, index=False)
dfi.export(combined, table_img_path)
print(f"📄 Saved table CSV: {table_csv_path}")
print(f"📸 Saved table image: {table_img_path}")

# === PLOT ===
scores = [best_row[m] for m in metrics]
x = np.arange(len(metrics))

plt.figure(figsize=(8, 5))
plt.bar(x, scores, color=COLOR_MAP.get("Original", "#4CAF50"), edgecolor="black")
for i, v in enumerate(scores):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.title(f"Best Original ({model_to_compare})")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
print(f"✅ Saved plot: {plot_path}")
