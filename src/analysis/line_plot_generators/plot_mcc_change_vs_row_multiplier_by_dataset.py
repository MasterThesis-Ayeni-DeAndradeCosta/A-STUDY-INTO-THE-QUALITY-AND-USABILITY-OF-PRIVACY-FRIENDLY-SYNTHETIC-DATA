import os
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

# ========== CONFIGURATION ==========

output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\synthetic\mccChange_vs_row_multiplier"
os.makedirs(output_root, exist_ok=True)

dataset_configs = {
    "Loan": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE",
            "epochs": 1000
        },
        "original_dataset": "Original"
    },
    "StudentPerformance": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE",
            "epochs": 1000
        },
        "original_dataset": "Original"
    },
    "BankMarketing": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE",
            "epochs": 50
        },
        "original_dataset": "Original"
    }
}

metric_column = "MCC"
x_axis = "row_multiplier"
change_column = "MCC_Change_Percent"

# ========== INITIALIZE ==========

plt.figure(figsize=(10, 6))
plotted_any = False
combined_excel_path = os.path.join(output_root, f"{change_column}_vs_{x_axis}_tables.xlsx")
excel_writer = pd.ExcelWriter(combined_excel_path, engine="xlsxwriter")

# ========== PROCESS EACH DATASET ==========

for dataset_name, config in dataset_configs.items():
    df = pd.read_excel(config["path"])

    # Extract original MCC for same model
    original_filter = (df["Dataset"] == config["original_dataset"]) & (df["Model"] == config["filters"]["Model"])
    original_df = df[original_filter]
    if original_df.empty:
        print(f"❌ No original data for {dataset_name} with model {config['filters']['Model']}")
        continue
    original_max_mcc = original_df[metric_column].max()

    # Filter synthetic TVAE data at fixed epochs
    df_filtered = df.copy()
    for key, val in config["filters"].items():
        df_filtered = df_filtered[df_filtered[key] == val]

    if df_filtered.empty or x_axis not in df_filtered.columns or metric_column not in df_filtered.columns:
        print(f"⚠️ No valid data for {dataset_name}, skipping...")
        continue

    # Clean and sort
    columns_to_save = [x_axis, metric_column, "Model", "Dataset", "epochs"]
    df_clean = df_filtered[columns_to_save].sort_values(by=x_axis)

    # Compute % MCC change vs original
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

    # Plot individual
    grouped = df_clean.groupby(x_axis)[change_column].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=dataset_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped.index, grouped.values, marker='o', label=dataset_name)
    ax.set_title(f"{dataset_name}: MCC Change (% vs Original) vs {x_axis}", fontsize=14)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("MCC Change (% vs Original)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    fig.tight_layout()

    indiv_plot_path = os.path.join(output_root, f"{dataset_name}_{change_column}_vs_{x_axis}_plot.png")
    fig.savefig(indiv_plot_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved: {dataset_name} MCC change (% vs Original) table + image + plot")
    plotted_any = True

# ========== COMBINED PLOT ==========

if plotted_any:
    plt.title(f"MCC Change (% vs Original) vs {x_axis} across Datasets", fontsize=14)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel("MCC Change (% vs Original)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(output_root, f"{change_column}_vs_{x_axis}_combined_plot.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.show()
    print(f"📊 Combined plot saved: {combined_plot_path}")
else:
    print("❌ No lines were plotted. Check filters or input files.")

# Finalize Excel
excel_writer.close()
print(f"📁 All Excel tables saved to: {combined_excel_path}")
