import os
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

def val_to_str(val):
    return str(val).replace(".", "p")

def generate_static_plots(df, variation_info_path, output_dir):
    if not os.path.exists(variation_info_path):
        print("⚠️ No variation_info.yaml found. Skipping plots.")
        return

    import yaml
    with open(variation_info_path, "r") as f:
        variation_info = yaml.safe_load(f)
    param_names = variation_info.get("varied_parameters", [])

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    PARAM_DATASET_MAP = {
        "epochs": ["CTGAN", "TVAE", "GaussianCopula"],
        "custom_generated_rows": ["CTGAN", "TVAE", "GaussianCopula"],
        "k_anonymity": ["Anonymous"],
        "l_diversity": ["Anonymous"],
        "suppression_limit": ["Anonymous"],
        "test_size": ["Original", "Anonymous", "CTGAN", "TVAE", "GaussianCopula"]
    }

    for metric in metrics:
        for xparam in param_names:
            sweep_values = sorted(df[xparam].dropna().unique())
            fixed_params = [p for p in param_names if p != xparam]
            sweep_dir = os.path.join(output_dir, metric.lower(), f"{metric.lower()}_vs_{xparam}")
            os.makedirs(sweep_dir, exist_ok=True)

            fixed_combos = list(product(*[sorted(df[p].dropna().unique()) for p in fixed_params])) or [()]

            for combo in fixed_combos:
                df_filtered = df.copy()
                suffix_parts = []
                for p, v in zip(fixed_params, combo):
                    df_filtered = df_filtered[df_filtered[p] == v]
                    suffix_parts.append(f"{p}={val_to_str(v)}")

                if xparam in PARAM_DATASET_MAP:
                    df_filtered = df_filtered[df_filtered["Dataset"].isin(PARAM_DATASET_MAP[xparam])]

                if df_filtered.empty:
                    continue

                plt.figure(figsize=(10, 6))
                for dataset in df_filtered["Dataset"].unique():
                    plt_df = df_filtered[df_filtered["Dataset"] == dataset].dropna(subset=[xparam, metric])
                    if not plt_df.empty:
                        plt.plot(plt_df[xparam], plt_df[metric], label=dataset, marker="o")

                plt.xlabel(xparam)
                plt.ylabel(metric)
                plt.title(f"{metric} vs {xparam} | {'; '.join(suffix_parts) if suffix_parts else 'all'}")
                plt.legend()
                plt.grid(True)
                plot_name = f"{'_'.join(suffix_parts) if suffix_parts else 'all'}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(sweep_dir, plot_name))
                plt.close()

    print(f"📊 Static plots generated in: {output_dir}")
