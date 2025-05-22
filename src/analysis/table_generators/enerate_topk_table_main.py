import os
import pandas as pd
import dataframe_image as dfi

# === CONFIG (EDIT THESE) ====================================================
input_excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx"
output_dir       = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\bar charts\topk"
output_name      = "loan_top5_ctgan_hybrid"
dataset_filter   = "Original"
model_filter     = "RandomForest"
sort_metric      = "MCC"
top_k            = 5
drop_columns     = ["suppressed_records", "suppression_percentage", "k_anonymity", "l_diversity", "suppression_limit"]
caption          = "Loan Top-5 — Original (RandomForest)"
# ============================================================================

# === LOAD DATA =============================================================
if not os.path.exists(input_excel_path):
    raise FileNotFoundError(f"❌ File not found: {input_excel_path}")

df = pd.read_excel(input_excel_path)
df = df[df["Model"] == model_filter]

if df.empty:
    raise ValueError(f"❌ No data found for model: {model_filter}")

df_filtered = df[df["Dataset"] == dataset_filter]
if df_filtered.empty:
    raise ValueError(f"❌ No rows with Dataset='{dataset_filter}' for model: {model_filter}")

topk = df_filtered.sort_values(sort_metric, ascending=False).head(top_k)

# Drop unnecessary columns
if drop_columns:
    topk = topk.drop(columns=[col for col in drop_columns if col in topk.columns])

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Export files
csv_path = os.path.join(output_dir, f"{output_name}.csv")
png_path = os.path.join(output_dir, f"{output_name}.png")

topk.to_csv(csv_path, index=False)
styled = topk.style.set_caption(caption)
dfi.export(styled, png_path)

print(f"✅ Top-{top_k} table exported:")
print(f"📄 CSV:  {csv_path}")
print(f"🖼️ PNG:  {png_path}")
