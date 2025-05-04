# import pandas as pd

# # === CONFIG ===
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\loan_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis"
# variant_field = "Dataset"

# # === MAIN FUNCTION ===
# def compare_hybrid_sources(df):
#     hybrid_variants = df[df[variant_field].str.contains("_HYBRID", na=False)][variant_field].unique()
#     comparison_tables = {}

#     for hybrid in hybrid_variants:
#         # Step 1: find best hybrid row
#         hybrid_df = df[df[variant_field] == hybrid]
#         best_row = hybrid_df.loc[hybrid_df["Average Metric"].idxmax()]
        
#         # Step 2: extract matching anonymized row
#         anon_row = df[
#             (df[variant_field] == "Anonymous") &
#             (df["k_anonymity"] == best_row["k_anonymity"]) &
#             (df["l_diversity"] == best_row["l_diversity"]) &
#             (df["suppression_limit"] == best_row["suppression_limit"])
#         ]

#         # Step 3: extract matching synthetic row
#         synth_name = hybrid.replace("_HYBRID", "")
#         synth_row = df[
#             (df[variant_field] == synth_name) &
#             (df["epochs"] == best_row["epochs"]) &
#             (df["row_multiplier"] == best_row["row_multiplier"]) &
#             (df["rows_generated_at_runtime"] == best_row["rows_generated_at_runtime"])
#         ]

#         # Step 4: original row (same model/dataset — just one entry usually)
#         original_row = df[df[variant_field] == "Original"]

#         # Step 5: concatenate results
#         combined_rows = pd.concat([
#             original_row.head(1).assign(Source="Original"),
#             anon_row.head(1).assign(Source="Anonymous"),
#             synth_row.head(1).assign(Source=synth_name),
#             best_row.to_frame().T.assign(Source=hybrid)
#         ])

#         combined_rows.insert(0, "Hybrid Group", hybrid)
#         comparison_tables[hybrid] = combined_rows.reset_index(drop=True)

#     return comparison_tables


# # === RUN ===
# if __name__ == "__main__":
#     df = pd.read_excel(excel_path)

#     tables = compare_hybrid_sources(df)

#     for hybrid_name, table in tables.items():
#         print(f"\n=== Comparison for {hybrid_name} ===")
#         print(table[["Source", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC",
#                      "k_anonymity", "l_diversity", "suppression_limit",
#                      "epochs", "row_multiplier",  "rows_generated_at_runtime"]])

#         # Optional: save to CSV
#         # table.to_csv(os.path.join(output_dir, f"{hybrid_name}_comparison.csv"), index=False)



import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

# === PATH FIX FOR COLOR_MAP IMPORT ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, parent_dir)

from color_palette import COLOR_MAP
import dataframe_image as dfi


# === CONFIG ===
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData"



variant_field = "Dataset"
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
model_to_compare = "RandomForest"

def plot_grouped_bar(dataframe, title, save_path):
    variants = dataframe["Source"]
    scores = dataframe[metrics].values.T
    x = np.arange(len(metrics))
    bar_width = 0.8 / len(variants)

    plt.figure(figsize=(12, 6))
    for i, row in enumerate(scores.T):
        color = COLOR_MAP.get(variants.iloc[i], "#CCCCCC")
        shift = i * bar_width - (bar_width * (len(variants)-1) / 2)
        plt.bar(x + shift, row, width=bar_width, label=variants.iloc[i], color=color, edgecolor="black")

        # Value labels
        for j, val in enumerate(row):
            plt.text(x[j] + shift, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    plt.xticks(x, metrics, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=10, frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved chart to: {save_path}")
    plt.close()

def process(df):
    hybrid_variants = df[df[variant_field].str.contains("_HYBRID", na=False)][variant_field].unique()

    for hybrid in hybrid_variants:
        hybrid_df = df[
            (df[variant_field] == hybrid) &
            (df["Model"] == model_to_compare)
        ]
        best_row = hybrid_df.loc[hybrid_df["Average Metric"].idxmax()]

        # Extract matching rows
        anon_row = df[
            (df["Model"] == model_to_compare) &
            (df[variant_field] == "Anonymous") &
            (df["k_anonymity"] == best_row["k_anonymity"]) &
            (df["l_diversity"] == best_row["l_diversity"]) &
            (df["suppression_limit"] == best_row["suppression_limit"])
        ]

        synth_dataset = hybrid.replace("_HYBRID", "")
        synth_row = df[
            (df["Model"] == model_to_compare) &
            (df[variant_field] == synth_dataset) &
            (df["epochs"] == best_row["epochs"]) &
            (df["row_multiplier"] == best_row["row_multiplier"])
        ]

        original_df = df[
            (df[variant_field] == "Original") &
            (df["Model"] == model_to_compare)
            ]

        original_row = original_df.loc[original_df["Average Metric"].idxmax()]
        original_row = original_row.to_frame().T.assign(Source="Original")


        # Combine rows
        combined = pd.concat([
            original_row.head(1).assign(Source="Original"),
            anon_row.head(1).assign(Source="Anonymous"),
            synth_row.head(1).assign(Source=synth_dataset),
            best_row.to_frame().T.assign(Source=hybrid)
        ])

        combined.reset_index(drop=True, inplace=True)

        # Save table
        csv_path = os.path.join(output_dir, f"{hybrid}_comparison.csv")
        combined.to_csv(csv_path, index=False)
        print(f"✅ Saved table to: {csv_path}")

        # Save the table as a PNG image too
        img_path = os.path.join(output_dir, f"{hybrid}_comparison_table.png")
        dfi.export(combined, img_path)
        print(f"📸 Saved table image to: {img_path}")

        # Save bar chart
        chart_path = os.path.join(output_dir, f"{hybrid}_bar_chart.png")
        plot_title = f"{hybrid} – Original vs Anonymous vs Synthetic vs Hybrid"
        plot_grouped_bar(combined, plot_title, chart_path)

if __name__ == "__main__":
    df = pd.read_excel(excel_path)
    process(df)
