import os
import sys
import matplotlib.pyplot as plt

# Dynamically fix the path (DO NOT CHANGE THIS)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, parent_dir)
from color_palette import COLOR_MAP


def plot_custom_metrics(model_name, dataset_name, metric_values, show_settings, save_dir, suffix=""):
    metrics = [m for m in show_settings if show_settings[m]]
    values = [metric_values[m] for m in metrics]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=COLOR_MAP.get("Original", "#2E86AB"), edgecolor="black")
    plt.ylim(0, 1)
    plt.title(f"{model_name} Performance on {dataset_name}", fontsize=14, weight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
                 ha='center', va='bottom', fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # File naming
    filename = f"{model_name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
    if suffix:
        filename += f"_{suffix}"
    filename += ".png"

    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Graph saved to: {full_path}")
    plt.show()


if __name__ == "__main__":
    model = "LogisticRegression"
    dataset = "Loan Dataset (Original)"

    metric_values = {
        "Accuracy": 0.8403,
        "Precision": 0.8431,
        "Recall": 0.8403,
        "F1": 0.8294,
        "AUC-ROC": 0.788,
        "LogLoss": 0.4495,
        "CohenKappa": 0.5839,
        "MCC": 0.6067,
        "Average Metric": 0.838275,
    }


    show_metrics = {
        "Accuracy": True,
        "Precision": True,
        "Recall": True,
        "F1": True,
        "AUC-ROC": True,
        "CohenKappa": False,
        "MCC": False,
        "Average Metric": False,
    }

    output_folder = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\results analysis\loan\bar charts"

    suffix = "best"

    plot_custom_metrics(model, dataset, metric_values, show_metrics, save_dir=output_folder, suffix=suffix)
