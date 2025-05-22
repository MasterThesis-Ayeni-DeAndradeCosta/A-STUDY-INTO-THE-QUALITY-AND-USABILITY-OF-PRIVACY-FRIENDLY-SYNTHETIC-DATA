import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\anonymous\ldiversity\l_diversity_lines"
os.makedirs(output_root, exist_ok=True)

dataset_configs = {
    # "Loan": {
    #     "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
    #     "model": "LogisticRegression",
    #     "suppression_limit": 0.1
    # },
    # "StudentPerformance": {
    #     "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
    #     "model": "RandomForest",
    #     "suppression_limit": 0.01
    # },
    # "BankMarketing": {
    #     "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
    #     "model": "RandomForest",
    #     "suppression_limit": 0.3
    # },
    "CensusIncome": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx",
        "model": "LogisticRegression",
        "suppression_limit": 0.3
    }
}

metric_col = "MCC"
x_col = "k_anonymity"

for dataset_name, config in dataset_configs.items():
    df = pd.read_excel(config["path"])
    
    # Original MCC
    original_mcc = df[
        (df["Model"] == config["model"]) & 
        (df["Dataset"] == "Original")
    ][metric_col].max()
    
    plt.figure(figsize=(8, 5))
    sns.set(style="whitegrid")
    plotted_any = False

    for l_val in [1, 2]:
        subset = df[
            (df["Model"] == config["model"]) &
            (df["Dataset"] == "Anonymous") &
            (df["l_diversity"] == l_val) &
            (df["suppression_limit"] == config["suppression_limit"])
        ]

        if subset.empty:
            print(f"⚠️ No data for {dataset_name} with ℓ={l_val}")
            continue

        grouped = subset.groupby(x_col)[metric_col].mean().sort_index()
        delta_mcc = ((grouped - original_mcc) / original_mcc * 100).round(2)

        label = f"ℓ = {l_val}"
        plt.plot(delta_mcc.index, delta_mcc.values, marker="o", label=label)
        plotted_any = True

        # Export to Excel
        export_df = pd.DataFrame({x_col: grouped.index, f"Δ MCC (%) (ℓ={l_val})": delta_mcc.values})
        excel_output_path = os.path.join(output_root, f"{dataset_name}_mcc_vs_k_l{l_val}.xlsx")
        export_df.to_excel(excel_output_path, index=False)

    if not plotted_any:
        continue

    plt.title(f"{dataset_name} – MCC Change vs $k$ (suppression limit = {int(config['suppression_limit']*100)}%)")
    plt.xlabel("$k$-Anonymity", fontsize=12)
    plt.ylabel("Δ MCC (%) from Original", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_root, f"{dataset_name}_mcc_vs_k_lines_by_l.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot and Excel files for {dataset_name}")
