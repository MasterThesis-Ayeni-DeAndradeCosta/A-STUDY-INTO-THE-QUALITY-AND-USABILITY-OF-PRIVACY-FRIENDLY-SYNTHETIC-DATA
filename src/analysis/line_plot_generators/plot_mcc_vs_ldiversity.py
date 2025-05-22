import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIG ===
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\plots\anonymization"
dataset_name = "Loan"
model_to_compare = "LogisticRegression"
selected_metric = "MCC"
k_fixed = 3
supp_fixed = 0.1

os.makedirs(output_dir, exist_ok=True)

# === LOAD DATA ===
df = pd.read_excel(excel_path)
df = df[(df["Model"] == model_to_compare) & (df["Dataset"] == "Anonymous")]

# Filter for fixed k and suppression
df = df[(df["k_anonymity"] == k_fixed) & (df["suppression_limit"] == supp_fixed)]

# Get original MCC for baseline
original_df = pd.read_excel(excel_path)
original_mcc = original_df[
    (original_df["Model"] == model_to_compare) & 
    (original_df["Dataset"] == "Original")
][selected_metric].max()

# Prepare plot data
df = df[["l_diversity", selected_metric]].dropna()
df = df.sort_values("l_diversity")
df["Δ MCC (%)"] = ((df[selected_metric] - original_mcc) / original_mcc * 100).round(2)

# === PLOT ===
plt.figure(figsize=(8, 5))
sns.set(style="whitegrid")
sns.lineplot(data=df, x="l_diversity", y="Δ MCC (%)", marker="o", linewidth=2, label="Loan")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("ℓ-Diversity", fontsize=12)
plt.ylabel("Δ MCC (%) from Original", fontsize=12)
plt.title(f"Loan – Effect of ℓ-Diversity on MCC (k = {k_fixed}, suppression limit = {supp_fixed:.0%})", fontsize=14)
plt.xticks(df["l_diversity"].unique())
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save plot
plot_path = os.path.join(output_dir, f"{dataset_name}_mcc_vs_ldiversity_k{k_fixed}_s{int(supp_fixed*100)}.png")
plt.savefig(plot_path, dpi=300)
plt.close()
