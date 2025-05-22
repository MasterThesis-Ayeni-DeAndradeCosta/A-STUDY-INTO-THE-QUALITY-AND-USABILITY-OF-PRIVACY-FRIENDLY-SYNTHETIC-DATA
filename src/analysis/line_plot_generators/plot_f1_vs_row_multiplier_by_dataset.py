import os
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

# ========== CONFIGURATION ==========

output_root = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\nice graphs\synthetic"
os.makedirs(output_root, exist_ok=True)

dataset_configs = {
    "Loan": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\loan_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE",
            "epochs": 1000  # Fixed value
        }
    },
    "StudentPerformance": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\studentPerformance\studentPerformance_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE",
            "epochs": 1000
        }
    },
    "BankMarketing": {
        "path": r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\bankMarketing\bankMarketing_combined_results.xlsx",
        "filters": {
            "Model": "RandomForest",
            "Dataset": "TVAE",
            "epochs": 50
        }
    }
}

metric_to_plot = "MCC"
x_axis = "row_multiplier"

# ========== INITIALIZE ==========

plt.figure(figsize=(10, 6))
plotted_any = False
combined_excel_path = os.path.join(output_root, f"{metric_to_plot}_vs_{x_axis}_tables.xlsx")
excel_writer = pd.ExcelWriter(combined_excel_path, engine="xlsxwriter")

# ========== PROCESS EACH DATASET ==========

for dataset_name, config in dataset_configs.items():
    df = pd.read_excel(config["path"])

    # Apply filters
    for key, val in config["filters"].items():
        df = df[df[key] == val]

    # Validate
    if df.empty or x_axis not in df.columns or metric_to_plot not in df.columns:
        print(f"⚠️ No valid data for {dataset_name}, skipping...")
        continue

    # Keep only relevant columns for clarity
    columns_to_save = [x_axis, metric_to_plot, "Model", "Dataset", "epochs"]
    df_clean = df[columns_to_save].sort_values(by=x_axis)

    # Save CSV and Excel
    csv_path = os.path.join(output_root, f"{dataset_name}_{metric_to_plot}_vs_{x_axis}.csv")
    excel_path = os.path.join(output_root, f"{dataset_name}_{metric_to_plot}_vs_{x_axis}.xlsx")
    df_clean.to_csv(csv_path, index=False)
    df_clean.to_excel(excel_path, index=False)
    df_clean.to_excel(excel_writer, sheet_name=dataset_name, index=False)

    # Save styled image table (matplotlib backend)
    styled = df_clean.style.set_caption(f"{dataset_name} — {metric_to_plot} vs {x_axis}")
    image_path = os.path.join(output_root, f"{dataset_name}_{metric_to_plot}_vs_{x_axis}.png")
    dfi.export(styled, image_path, table_conversion='matplotlib')

    # Plot individual line for combined plot
    grouped = df_clean.groupby(x_axis)[metric_to_plot].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=dataset_name)

    # Save individual plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped.index, grouped.values, marker='o', label=dataset_name)
    ax.set_title(f"{dataset_name}: {metric_to_plot} vs {x_axis}", fontsize=14)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(metric_to_plot)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    indiv_plot_path = os.path.join(output_root, f"{dataset_name}_{metric_to_plot}_vs_{x_axis}_plot.png")
    fig.savefig(indiv_plot_path, dpi=300)
    plt.close(fig)

    print(f"✅ Saved: {dataset_name} table + image + plot")
    plotted_any = True

# ========== COMBINED PLOT ==========

if plotted_any:
    plt.title(f"{metric_to_plot} vs {x_axis} across Datasets", fontsize=14)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel(metric_to_plot, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    combined_plot_path = os.path.join(output_root, f"{metric_to_plot}_vs_{x_axis}_combined_plot.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.show()
    print(f"📊 Combined plot saved: {combined_plot_path}")
else:
    print("❌ No lines were plotted. Check filters or input files.")

# Finalize master Excel
excel_writer.close()
print(f"📁 All Excel tables saved to: {combined_excel_path}")
