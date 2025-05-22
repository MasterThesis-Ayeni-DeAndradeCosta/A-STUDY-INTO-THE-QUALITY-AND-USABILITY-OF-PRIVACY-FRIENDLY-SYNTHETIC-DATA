"""
select_best_models.py
Pick the best-performing row for every distinct Model
after applying the same simple filter logic used in
plot_metrics_by_parameter.py, then export:

  • <suffix>_best_models_table.csv
  • <suffix>_best_models_table.png

Author: you
"""
import os
import pandas as pd
import dataframe_image as dfi   # ← already installed for the plotting script

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# (copy/paste paths & filters from plot_metrics_by_parameter.py)
#loan
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\tables"
# dataset_name = "loan"

#studentPerformance
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\plots"
dataset_name = "StudentPerformance"

# bankMarketing
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing"
# dataset_name = "bankMarketing"

# censusIncome
# excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx"
# output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome"
# dataset_name = "censusIncome"



# same key/value logic as in the plotting file
filters = {
    "Dataset": "Original",      # "Anonymous", "Synthetic", "Hybrid", …
    # "Model":   "RandomForest" # add more filters if you want
}

primary_metric = "MCC"          # Accuracy, Precision, … anything in your sheet
# ─────────────────────────────────────────────────────────────────────────────

# derive the same suffix convention as the plotting file
suffix = f"{dataset_name.lower()}_{filters.get('Dataset', 'all').lower()}_{primary_metric.lower()}"

def select_best_models(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(f"❌ Metric '{metric}' not found in DataFrame columns.")
    best_rows = (
        df.loc[df.groupby("Model")[metric].idxmax()]
        .sort_values(metric, ascending=False)
        .reset_index(drop=True)
    )
    best_rows.index += 1
    return best_rows
def export_table(df_best: pd.DataFrame, save_dir: str, suff: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{suff}_best_models_table.csv")
    img_path = os.path.join(save_dir, f"{suff}_best_models_table.png")
    df_best.to_csv(csv_path, index=False)

    # For image, remove extra columns if needed
    png_df = df_best.copy()
    if "Average Metric" in png_df.columns:
        png_df = png_df.drop(columns=["Average Metric"])
    if "MCC" in png_df.columns:
        mcc_index = png_df.columns.get_loc("MCC")
        png_df = png_df.iloc[:, :mcc_index + 1]

    dfi.export(png_df.style.set_caption("Best-Performing Models"), img_path)
    print(f"📄 CSV saved to: {csv_path}")
    print(f"🖼️ PNG saved to: {img_path}")

def main() -> None:
    df = pd.read_excel(excel_path)

    for col, val in filters.items():
        df = df[df[col] == val]

    if df.empty:
        print("❌ No data matches the filter criteria.")
        return

    df_best = select_best_models(df, primary_metric)
    print("🏆 Best models (highest → lowest):")
    print(df_best[["Model", "Dataset", primary_metric]])
    export_table(df_best, output_dir, suffix)

if __name__ == "__main__":
    main()
