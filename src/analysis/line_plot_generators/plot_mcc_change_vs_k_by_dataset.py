import os
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

# ========== CONFIGURATION ==========

# output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\anonymous\mccChange"

# output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\hybrid\mccChange_vs_k_anonymity"
output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\fixed graphs"

os.makedirs(output_root, exist_ok=True)

# dataset_configs = {
#     "Loan": {
#         "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
#         "filters": {
#             "Model": "LogisticRegression",
#             "Dataset": "Anonymous",
#             "l_diversity": 2,
#             "suppression_limit": 0.1
#         },
#         "original_dataset": "Original"
#     },
#     "StudentPerformance": {
#         "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
#         "filters": {
#             "Model": "RandomForest",
#             "Dataset": "Anonymous",
#             "l_diversity": 2,
#             "suppression_limit": 0.2
#         },
#         "original_dataset": "Original"
#     },
#     "BankMarketing": {
#         "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
#         "filters": {
#             "Model": "RandomForest",
#             "Dataset": "Anonymous",
#             "l_diversity": 2,
#             "suppression_limit": 0.3
#         },
#         "original_dataset": "Original"
#     },
#     "CensusIncome": {
#         "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx",
#         "filters": {
#             "Model": "LogisticRegression",
#             "Dataset": "Anonymous",
#             "l_diversity": 1,
#             "suppression_limit": 0.3
#         },
#         "original_dataset": "Original"
#     }
# }


dataset_configs = {
    "Loan": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
        "filters": {
            "Model": "LogisticRegression",
            "Dataset": "GaussianCopula_HYBRID",
            "l_diversity": 15,
            "suppression_limit": 0.1
        },
        "original_dataset": "Original"
    },
    "StudentPerformance": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE_HYBRID",
            "l_diversity": 1,
            "suppression_limit": 0.2
        },
        "original_dataset": "Original"
    },
    "BankMarketing": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "CTGAN_HYBRID",
            "l_diversity": 2,
            "suppression_limit": 0.3
        },
        "original_dataset": "Original"
    },
    "CensusIncome": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx",
        "filters": {
            "Model": "LogisticRegression",
            "Dataset": "TVAE_HYBRID",
            "l_diversity": 2,
            "suppression_limit": 0.3
        },
        "original_dataset": "Original"
    }
}





metric_column = "MCC"
x_axis = "k_anonymity"
change_column = "MCC_Change_Percent"

# ========== INITIALIZE ==========

plt.figure(figsize=(10, 6))
plotted_any = False
combined_excel_path = os.path.join(output_root, f"{change_column}_vs_{x_axis}_tables.xlsx")
excel_writer = pd.ExcelWriter(combined_excel_path, engine="xlsxwriter")

# ========== PROCESS EACH DATASET ==========

for dataset_name, config in dataset_configs.items():
    df = pd.read_excel(config["path"])

    # Extract original MCC for the same model
    original_filter = (df["Dataset"] == config["original_dataset"]) & (df["Model"] == config["filters"]["Model"])
    original_df = df[original_filter]

    if original_df.empty:
        print(f"❌ No original data for {dataset_name} with model {config['filters']['Model']}")
        continue

    original_max_mcc = original_df[metric_column].max()

    # Apply anonymized filters
    df_filtered = df.copy()
    for key, val in config["filters"].items():
        df_filtered = df_filtered[df_filtered[key] == val]

    if df_filtered.empty or x_axis not in df_filtered.columns or metric_column not in df_filtered.columns:
        print(f"⚠️ No valid anonymized data for {dataset_name}, skipping...")
        continue

    # Clean and sort
    columns_to_save = [x_axis, metric_column, "Model", "Dataset", "l_diversity", "suppression_limit"]
    df_clean = df_filtered[columns_to_save].sort_values(by=x_axis)
    df_clean = df_clean[df_clean[x_axis] <= 50]  # Cap k

    # Calculate % MCC change vs original
    df_clean[change_column] = ((df_clean[metric_column] - original_max_mcc) / original_max_mcc) * 100

    # Save CSV, Excel, styled image
    csv_path = os.path.join(output_root, f"{dataset_name}_{change_column}_vs_{x_axis}.csv")
    excel_path = os.path.join(output_root, f"{dataset_name}_{change_column}_vs_{x_axis}.xlsx")
    df_clean.to_csv(csv_path, index=False)
    df_clean.to_excel(excel_path, index=False)
    df_clean.to_excel(excel_writer, sheet_name=dataset_name, index=False)

    styled = df_clean.style.set_caption(f"{dataset_name} — MCC Change (% vs Original) vs {x_axis}")
    image_path = os.path.join(output_root, f"{dataset_name}_{change_column}_vs_{x_axis}.png")
    dfi.export(styled, image_path, table_conversion='matplotlib')

    # Plot line
    grouped = df_clean.groupby(x_axis)[change_column].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=dataset_name)

    # Individual plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped.index, grouped.values, marker='o', label=dataset_name)
    ax.set_title(f"{dataset_name}: MCC Change vs. $k$-anonymity level", fontsize=14)
    ax.set_xlabel(r"$k$-anonymity level")
    ax.set_ylabel("MCC Change (% vs Original)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    # ✅ CHANGED: Add baseline to ax, not plt
    ax.axhline(0, color='black', linestyle='--', linewidth=1, label="Original")

    # ✅ CHANGED: Use ax.legend to affect the subplot
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()

    indiv_plot_path = os.path.join(output_root, f"{dataset_name}_{change_column}_vs_{x_axis}_plot.png")
    fig.savefig(indiv_plot_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved: {dataset_name} MCC change (% vs Original) table + image + plot")
    plotted_any = True

# ========== COMBINED PLOT ==========

if plotted_any:
    plt.title(r"MCC Change vs. $k$-anonymity level", fontsize=14)
    plt.xlabel(r"$k$-anonymity level", fontsize=12)
    plt.ylabel("MCC Change (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # ✅ Add baseline with label
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label="Original")

    # # ✅ Move legend away from data
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # plt.tight_layout(rect=[0, 0, 0.85, 1])  # ✅ Ensure room for legend on right

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Leave room at bottom for legend


    combined_plot_path = os.path.join(output_root, f"{change_column}_vs_{x_axis}_combined_plot.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.show()
    print(f"📊 Combined plot saved: {combined_plot_path}")
else:
    print("❌ No lines were plotted. Check filters or input files.")

# Finalize Excel
excel_writer.close()
print(f"📁 All Excel tables saved to: {combined_excel_path}")
