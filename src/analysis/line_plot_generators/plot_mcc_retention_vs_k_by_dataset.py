import os
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

# ========== CONFIGURATION ==========

output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\anonymous\mccretention"
os.makedirs(output_root, exist_ok=True)

dataset_configs = {
    "Loan": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
        "filters": {
            "Model": "LogisticRegression",
            "Dataset": "Anonymous",
            "l_diversity": 2,
            "suppression_limit": 0.1
        }
    },
    "StudentPerformance": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "Anonymous",
            "l_diversity": 2,
            "suppression_limit": 0.2
        }
    },
    "BankMarketing": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "Anonymous",
            "l_diversity": 2,
            "suppression_limit": 0.3
        }
    },
    "CensusIncome": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\censusIncome\censusIncome_combined_results.xlsx",
        "filters": {
            "Model": "LogisticRegression",
            "Dataset": "Anonymous",
            "l_diversity": 1,
            "suppression_limit": 0.3
        }
    }
}

metric_column = "MCC"
x_axis = "k_anonymity"
retention_column = "MCC_Retention"

# ========== INITIALIZE ==========

plt.figure(figsize=(10, 6))
plotted_any = False
combined_excel_path = os.path.join(output_root, f"{retention_column}_vs_{x_axis}_tables.xlsx")
excel_writer = pd.ExcelWriter(combined_excel_path, engine="xlsxwriter")

# ========== PROCESS EACH DATASET ==========

for dataset_name, config in dataset_configs.items():
    df = pd.read_excel(config["path"])

    # Apply filters
    for key, val in config["filters"].items():
        df = df[df[key] == val]

    # Validate
    if df.empty or x_axis not in df.columns or metric_column not in df.columns:
        print(f"⚠️ No valid data for {dataset_name}, skipping...")
        continue

    # Keep only relevant columns
    columns_to_save = [x_axis, metric_column, "Model", "Dataset", "l_diversity", "suppression_limit"]
    df_clean = df[columns_to_save].sort_values(by=x_axis)

    # Cap k_anonymity to 50
    df_clean = df_clean[df_clean[x_axis] <= 50]

    # Compute MCC retention relative to best
    max_mcc = df_clean[metric_column].max()
    df_clean[retention_column] = df_clean[metric_column] / max_mcc

    # Save CSV, Excel, styled image
    csv_path = os.path.join(output_root, f"{dataset_name}_{retention_column}_vs_{x_axis}.csv")
    excel_path = os.path.join(output_root, f"{dataset_name}_{retention_column}_vs_{x_axis}.xlsx")
    df_clean.to_csv(csv_path, index=False)
    df_clean.to_excel(excel_path, index=False)
    df_clean.to_excel(excel_writer, sheet_name=dataset_name, index=False)

    styled = df_clean.style.set_caption(f"{dataset_name} — Utility Retention (MCC%) vs {x_axis}")
    image_path = os.path.join(output_root, f"{dataset_name}_{retention_column}_vs_{x_axis}.png")
    dfi.export(styled, image_path, table_conversion='matplotlib')

    # Plot line
    grouped = df_clean.groupby(x_axis)[retention_column].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=dataset_name)

    # Individual plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped.index, grouped.values, marker='o', label=dataset_name)
    ax.set_title(f"{dataset_name}: Utility Retention (MCC%) vs {x_axis}", fontsize=14)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Utility Retention (MCC % of best)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    indiv_plot_path = os.path.join(output_root, f"{dataset_name}_{retention_column}_vs_{x_axis}_plot.png")
    fig.savefig(indiv_plot_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved: {dataset_name} retention table + image + plot")
    plotted_any = True

# ========== COMBINED PLOT ==========

if plotted_any:
    plt.title(f"Utility Retention (MCC%) vs {x_axis} across Datasets", fontsize=14)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel("Utility Retention (MCC % of best)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(output_root, f"{retention_column}_vs_{x_axis}_combined_plot.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.show()
    print(f"📊 Combined plot saved: {combined_plot_path}")
else:
    print("❌ No lines were plotted. Check filters or input files.")

# Finalize master Excel
excel_writer.close()
print(f"📁 All Excel tables saved to: {combined_excel_path}")