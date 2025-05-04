import os
import sys
import matplotlib.pyplot as plt

# Dynamically fix the path (DO NOT CHANGE THIS)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, parent_dir)
from color_palette import COLOR_MAP

def plot_comparison(metric_dict1, metric_dict2, label1, label2, title, save_dir, suffix="", k=None, l=None, suppression=None):
    metrics = list(metric_dict1.keys())
    values1 = [metric_dict1[m] for m in metrics]
    values2 = [metric_dict2[m] for m in metrics]

    x = range(len(metrics))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - bar_width/2 for i in x], values1, width=bar_width,
            label=label1, color=COLOR_MAP.get(label1, "#2E86AB"), edgecolor="black")
    plt.bar([i + bar_width/2 for i in x], values2, width=bar_width,
            label=label2, color=COLOR_MAP.get(label2, "#E27D60"), edgecolor="black")

    plt.xticks(ticks=x, labels=metrics, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12)
    plt.suptitle(title, fontsize=14, weight="bold")

    if k is not None and l is not None and suppression is not None:
        subtitle = f"Anonymization settings: k={k}, l={l}, suppression={suppression}"
        plt.title(subtitle, fontsize=10)

    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    # Add value labels
    for i, (v1, v2) in enumerate(zip(values1, values2)):
        plt.text(i - bar_width/2, v1 + 0.01, f"{v1:.3f}", ha='center', va='bottom', fontsize=9)
        plt.text(i + bar_width/2, v2 + 0.01, f"{v2:.3f}", ha='center', va='bottom', fontsize=9)


    filename = f"{label1.lower()}_vs_{label2.lower()}_{suffix}.png"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Graph saved to: {full_path}")
    plt.show()


if __name__ == "__main__":
    # === Tweak values here only ===
    output_folder = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\Semester 2\writing the thesis\graphs\loan\anonymous"
    suffix = "best_anonymous"

    title = "Random Forest Performance: Original vs Anonymized (Loan Dataset)"

    label1 = "Original"
    label2 = "Anonymous"

    original_metrics = {
        "Accuracy": 0.854,
        "Precision": 0.860,
        "Recall": 0.854,
        "F1": 0.844,
        "AUC-ROC": 0.853,
    }

    anonymized_metrics = {
    "Accuracy": 0.6389,
    "Precision": 0.6297,
    "Recall": 0.6389,
    "F1": 0.6338,
    "AUC-ROC": 0.5728,
}


    anonymization_k = 5
    anonymization_l = 2
    suppression_limit = 0.08


    # === Call plotting function ===
    plot_comparison(
        original_metrics,
        anonymized_metrics,
        label1,
        label2,
        title,
        save_dir=output_folder,
        suffix=suffix,
        k=anonymization_k,
        l=anonymization_l,
        suppression=suppression_limit
    )
