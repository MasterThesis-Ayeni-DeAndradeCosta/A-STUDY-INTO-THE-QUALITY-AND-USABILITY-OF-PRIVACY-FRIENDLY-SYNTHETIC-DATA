"""
loan_best_anonymized.py
────────────────────────────────────────────────────────────────────────────
• Reads the combined results for the *Loan* dataset
• Finds … 
    – the single best Anonymous row (highest F1)
    – the matching best Original row for **that** model
    – the Top-5 Anonymous rows (F1 ranking)
• Exports three neat tables  (CSV + PNG)
• Plots the usual bar-chart    (Original vs best Anonymous)
"""

import os, pandas as pd, dataframe_image as dfi
import matplotlib.pyplot as plt      # only for the bar-chart
from color_palette import COLOR_MAP 
import numpy as np

# ── loan PATHS ──────────────────────────────────────────────────────────────────
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx"
# out_dir    = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\bar charts\anonymous"

# ── studentPerformance PATHS ────────────────────────────────────────────────────
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx"
out_dir    = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\plots\anonymous"

# ── bankMarketing PATHS ────────────────────────────────────────────────────────
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx"
# out_dir    = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\anonymous"



# === PATHS ===
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx"
# out_dir    = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\anonymous"


os.makedirs(out_dir, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────────────────
dataset_name     = "StudentPerformance"  # "Loan", "StudentPerformance", "BankMarketing"
model_to_compare   = "RandomForest"  # "LogisticRegression", "RandomForest", …
primary_metric   = "MCC"     # change to "Accuracy", … if needed
top_k            = 5
metrics_for_plot = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
drop_cols          = ["epochs", "row_multiplier", "rows_generated_at_runtime"]

# === LOAD & FILTER ===
df = pd.read_excel(excel_path)
anon_sorted = df[(df["Dataset"] == "Anonymous") & (df["Model"] == model_to_compare)].sort_values(primary_metric, ascending=False)

if anon_sorted.empty:
    raise SystemExit(f"❌ No Anonymous rows found for model '{model_to_compare}'.")

best_anon = anon_sorted.iloc[[0]]

orig_rows = df[(df["Dataset"] == "Original") & (df["Model"] == model_to_compare)]
if orig_rows.empty:
    raise SystemExit(f"❌ No Original rows found for model '{model_to_compare}'.")

best_orig = orig_rows.loc[orig_rows[primary_metric].idxmax()].to_frame().T

# === EXPORT TABLES ===
def save_table(table: pd.DataFrame, stub: str) -> None:
    csv = os.path.join(out_dir, f"{stub}.csv")
    png = os.path.join(out_dir, f"{stub}.png")
    table_cleaned = table.drop(columns=[c for c in drop_cols if c in table.columns])
    table_cleaned.to_csv(csv, index=False)
    styled = table_cleaned.copy()
    styled.index = range(1, len(styled) + 1)
    dfi.export(styled.style.set_caption(stub.replace('_', ' ').title()), png)
    print(f"📄 {csv}\n🖼️  {png}")

save_table(best_anon, f"{dataset_name}_anonymous_best_{model_to_compare}")
save_table(pd.concat([best_orig, best_anon], ignore_index=True),
           f"{dataset_name}_original_vs_best_anonymous_{model_to_compare}")

# === BAR CHART ===
x = np.arange(len(metrics_for_plot))
bar_width = 0.35
orig_vals = [float(best_orig.iloc[0][m]) for m in metrics_for_plot]
anon_vals = [float(best_anon.iloc[0][m]) for m in metrics_for_plot]

plt.figure(figsize=(10, 5))
plt.bar(x - bar_width/2, orig_vals, width=bar_width,
        color=COLOR_MAP.get("Original", "#2E86AB"),
        label="Original", edgecolor="black")
plt.bar(x + bar_width/2, anon_vals, width=bar_width,
        color=COLOR_MAP.get("Anonymous", "#E27D60"),
        label="Anonymous", edgecolor="black")

for i, (v1, v2) in enumerate(zip(orig_vals, anon_vals)):
    plt.text(i - bar_width/2, v1 + 0.01, f"{v1:.3f}", ha="center", va="bottom", fontsize=8)
    plt.text(i + bar_width/2, v2 + 0.01, f"{v2:.3f}", ha="center", va="bottom", fontsize=8)

plt.xticks(x, metrics_for_plot, rotation=45)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title(f"{model_to_compare}: Original vs Anonymous ({dataset_name.title()}) — ranked by {primary_metric}")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.tight_layout(rect=[0, 0.15, 1, 1])

chart_path = os.path.join(out_dir, f"{dataset_name}_bar_original_vs_anonymous_{model_to_compare}.png")
plt.savefig(chart_path, dpi=300)
plt.close()
print(f"📊 Bar-chart saved: {chart_path}")