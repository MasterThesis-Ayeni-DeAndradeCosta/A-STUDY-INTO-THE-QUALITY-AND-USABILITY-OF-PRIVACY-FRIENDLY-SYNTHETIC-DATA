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
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\hybridTables\loan"
# dataset_name = "Loan"
# os.makedirs(output_dir, exist_ok=True)

# studentPerformance paths (active)
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\hybridTables\studentPerformance"
# dataset_name = "studentPerformance"
# os.makedirs(output_dir, exist_ok=True)

# bankMarketing paths
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\hybridTables\bankMarketing"
dataset_name = "bankMarketing"
os.makedirs(output_dir, exist_ok=True)

# censusIncome
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\hybridTables\censusIncome"
# dataset_name = "censusIncome"
# os.makedirs(output_dir, exist_ok=True)

# === CONFIG ===
model_to_compare = "RandomForest" # "LogisticRegression", "RandomForest", …
selected_metric = "MCC"
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

# === COMPARE MATCHED CHAINS ===
for synth in synth_bases:
    hybrid_label = f"{synth}_HYBRID"
    hybrid_df = df[df["Dataset"] == hybrid_label]
    if hybrid_df.empty:
        print(f"⚠️ No rows for {hybrid_label}, skipping.")
        continue

    # Find best hybrid row
    best_hybrid = hybrid_df.loc[hybrid_df[selected_metric].idxmax()]

    # Extract config values
    k, l, s = best_hybrid["k_anonymity"], best_hybrid["l_diversity"], best_hybrid["suppression_limit"]
    epochs, multiplier = best_hybrid["epochs"], best_hybrid["row_multiplier"]

    # Match Anonymous row with same k, l, s
    anon_df = df[
        (df["Dataset"] == "Anonymous") &
        (df["k_anonymity"] == k) &
        (df["l_diversity"] == l) &
        (df["suppression_limit"] == s)
    ]
    best_anon = anon_df[anon_df[selected_metric] == anon_df[selected_metric].max()] if not anon_df.empty else None

    # Match Synthetic row with same epochs, multiplier
    synth_df = df[
        (df["Dataset"] == synth) &
        (df["epochs"] == epochs) &
        (df["row_multiplier"] == multiplier)
    ]
    best_synth = synth_df[synth_df[selected_metric] == synth_df[selected_metric].max()] if not synth_df.empty else None

    # Original
    original_df = df[df["Dataset"] == "Original"]
    if original_df.empty:
        continue
    best_original = original_df.loc[original_df[selected_metric].idxmax()]

    step_rows = [("Original", best_original)]
    if best_anon is not None and not best_anon.empty:
        step_rows.append(("Anonymous", best_anon.iloc[0]))
    else:
        print(f"⚠️ No matching Anonymous row for {synth}. Skipping.")
        continue

    if best_synth is not None and not best_synth.empty:
        step_rows.append((synth, best_synth.iloc[0]))
    else:
        print(f"⚠️ No matching Synthetic row for {synth}. Skipping.")
        continue

    step_rows.append((hybrid_label, best_hybrid))

    # Build DataFrame
    full_table = pd.DataFrame([r[1] for r in step_rows])
    full_table.insert(0, "Source", [r[0] for r in step_rows])

    front_cols = ["Source", "Model", "Dataset"] + metrics + [selected_metric, "AUC-ROC"] + important_config_cols
    full_table = full_table[[col for col in front_cols if col in full_table.columns] + 
                            [col for col in full_table.columns if col not in front_cols]]

    base = synth.lower()
    clean_table = full_table.drop(columns=[c for c in full_table.columns if c not in metrics + ["Source", "Model", "Dataset", selected_metric, "AUC-ROC"]])

    clean_csv = os.path.join(output_dir, f"{dataset_name}_chain_matched_{base}.csv")
    clean_png = os.path.join(output_dir, f"{dataset_name}_chain_matched_{base}.png")
    full_csv = os.path.join(output_dir, f"{dataset_name}_chain_matched_{base}_full.csv")
    full_png = os.path.join(output_dir, f"{dataset_name}_chain_matched_{base}_full.png")

    clean_table.to_csv(clean_csv, index=False)
    dfi.export(clean_table.style.set_caption(f"{dataset_name.title()} – Matched Chain (Metrics Only) – {synth}"), clean_png)
    full_table.to_csv(full_csv, index=False)
    dfi.export(full_table.style.set_caption(f"{dataset_name.title()} – Matched Chain (Full Config) – {synth}"), full_png)
