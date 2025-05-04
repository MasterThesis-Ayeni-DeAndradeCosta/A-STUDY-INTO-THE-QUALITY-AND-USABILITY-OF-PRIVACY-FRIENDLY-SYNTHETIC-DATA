import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Fix path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, parent_dir)
from color_palette import COLOR_MAP


def plot_grouped_metrics(metrics_dict, params_dict, save_path, dataset_name):
    variants = list(metrics_dict.keys())                         # e.g. Original, CTGAN, …
    metrics  = list(next(iter(metrics_dict.values())).keys())     # e.g. Accuracy, F1, …

    n_variants = len(variants)
    x          = np.arange(len(metrics))
    bar_width  = 0.15

    plt.figure(figsize=(12, 7))

    # ───────────────────────────────────────────────────────── plot bars
    for i, variant in enumerate(variants):
        scores = [metrics_dict[variant][m] for m in metrics]
        plt.bar(
            x + i * bar_width,
            scores,
            width=bar_width,
            label=variant,
            color=COLOR_MAP.get(variant, "#CCCCCC"),
            edgecolor="black",
        )

        # value labels
        for j, score in enumerate(scores):
            plt.text(
                x[j] + i * bar_width,
                score + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # ───────────────────────────────────────────────────────── cosmetics
    plt.xticks(x + bar_width * (n_variants - 1) / 2, metrics, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)

    # Title
    plt.title(
        f"Random Forest Performance on {dataset_name}: Original vs Synthetic",
        fontsize=14, weight="bold",
    )

    # Legend centered below chart
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=n_variants,
        fontsize=10,
        frameon=False,
    )

    # Footnote with synthesizer parameters
    param_note = " | ".join(
        f"{v}: epochs={p['epochs']}, mult={p['w_multiplier']}"
        for v, p in params_dict.items() if v != "Original"
    )
    plt.figtext(0.5, -0.28, param_note, wrap=True, ha="center", fontsize=9)

    # Reserve vertical space for both legend and footnote
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Save (no bbox_inches, avoids layout override)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    output_path = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\Semester 2\writing the thesis\graphs\loan\synthetic\original_vs_all_synthetic_grouped.png"
    dataset_name = "Loan Dataset"

    metrics_dict = {
    "Original": {
        "Accuracy": 0.854,
        "Precision": 0.860,
        "Recall": 0.854,
        "F1": 0.844,
        "AUC-ROC": 0.853,
    },
    "CTGAN": {
        "Accuracy": 0.4028,
        "Precision": 0.4651,
        "Recall": 0.4028,
        "F1": 0.4256,
        "AUC-ROC": 0.2959,
    },
    "TVAE": {
        "Accuracy": 0.6667,
        "Precision": 0.4762,
        "Recall": 0.6667,
        "F1": 0.5556,
        "AUC-ROC": 0.5085,
    },
    "GaussianCopula": {
        "Accuracy": 0.6875,
        "Precision": 0.5845,
        "Recall": 0.6875,
        "F1": 0.5778,
        "AUC-ROC": 0.7324,
    }
}

    params_dict = {
        "CTGAN": {
            "epochs": 300,
            "w_multiplier": 3,
            "rows_generated": 1008
        },
        "TVAE": {
            "epochs": 400,
            "w_multiplier": 3,
            "rows_generated": 1008
        },
        "GaussianCopula": {
            "epochs": 300,
            "w_multiplier": 8,
            "rows_generated": 336
        }
    }


    plot_grouped_metrics(metrics_dict, params_dict, output_path, dataset_name)
