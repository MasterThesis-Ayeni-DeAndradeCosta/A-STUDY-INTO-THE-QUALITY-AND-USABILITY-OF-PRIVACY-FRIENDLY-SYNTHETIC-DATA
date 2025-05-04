import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import dataframe_image as dfi


# === CONFIGURATION (Only edit this block) ===
excel_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\MasterThesisCode\outputs\batch\crimeData_batch_2025-04-20_21-30-12\batch_analysis\combined_results.xlsx"
output_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\saved outputs\sorted analysis\crimeData\line_plots"
dataset_display_name = "Crime Data"
model_to_compare = "RandomForest"
x_axis = "suppression_percentage"


filters = {
    "Model": "RandomForest",
    "Dataset": "Anonymous",
    "k_anonymity": 5,
    "l_diversity": 2,
}



show_metrics = {
    "Accuracy": True,
    "Precision": True,
    "Recall": True,
    "F1": True,
    "AUC-ROC": True,
    "Average Metric": False,
    "LogLoss": False,
    "CohenKappa": False,
    "MCC": False,
}


suffix = f"{filters['Dataset'].lower()}_{x_axis}"
# =============================================

def plot_metrics(df, x_axis, show_metrics, save_dir, suffix):
    metrics = [m for m, show in show_metrics.items() if show]
    x_vals = sorted(df[x_axis].unique())
    metrics_dict = {metric: [] for metric in metrics}

    plt.figure(figsize=(10, 6))
    for metric in metrics:
        y_vals = []
        for x in x_vals:
            group = df[df[x_axis] == x]
            if group.empty or metric not in group.columns:
                y_vals.append(None)
            else:
                y_vals.append(group[metric].mean())
        metrics_dict[metric] = y_vals  # <-- this was missing
        plt.plot(x_vals, y_vals, marker='o', label=metric)

    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel("Score", fontsize=12)

    # Title and subtitle logic
    #main_title = f"{' vs '.join(metrics)} by {x_axis}"
    main_title = f"Metrics vs {x_axis} ({dataset_display_name})"

    param_str = ', '.join([f"{k}={v}" for k, v in filters.items() if k != x_axis])
    plt.suptitle(main_title, fontsize=14, weight='bold')
    plt.title(f"Fixed parameters: {param_str}", fontsize=10)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1)
    plt.tight_layout()

    filename = f"metrics_vs_{x_axis}_{suffix}.png"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Saved to {full_path}")

    # Export CSV + image of metrics table
    df_table = pd.DataFrame(metrics_dict, index=x_vals)
    df_table.index.name = x_axis

    csv_path = os.path.join(save_dir, f"{suffix}_table.csv")
    img_path = os.path.join(save_dir, f"{suffix}_table.png")
    df_table.to_csv(csv_path)
    dfi.export(df_table, img_path)

    print(f"📄 Table CSV saved to: {csv_path}")
    print(f"📸 Table image saved to: {img_path}")

    
    
    plt.show()

def main():
    df = pd.read_excel(excel_path)

    for col, val in filters.items():
        df = df[df[col] == val]

    if df.empty:
        print("❌ No data matches the filter criteria.")
        return

    plot_metrics(df, x_axis, show_metrics, output_dir, suffix)

if __name__ == "__main__":
    main()

