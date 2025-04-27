import os
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import yaml

def val_to_str(val):
    return str(val).replace(".", "p")

def generate_static_plots(df, variation_info_path, output_dir):
    """
    For each xparam, creates a folder e.g. 'batch_analysis/k_anonymity/'.
    Inside that folder, subfolders for each metric: 'Accuracy/', 'Precision/', etc.
    Then inside each metric folder, subfolders for each model: 'DecisionTree/', 'KNN/', etc.
    Inside each model folder:
      - a PNG plot
      - a CSV file of the exact data used
    showing the lines per Dataset (CTGAN, TVAE, etc.).
    """
    if not os.path.exists(variation_info_path):
        print("⚠️ No variation_info.yaml found. Skipping plots.")
        return

    with open(variation_info_path, "r") as f:
        var_info = yaml.safe_load(f)

    xparams = var_info.get("varied_parameters", [])
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    metrics = [m for m in metrics if m in df.columns]

    PARAM_DATASET_MAP = {
        "k_anonymity": ["Anonymous"],
        "l_diversity": ["Anonymous"],
        "suppression_limit": ["Anonymous"],
        "epochs": ["CTGAN", "TVAE", "GaussianCopula", "Hybrid"],
        "custom_generated_rows": ["CTGAN", "TVAE", "GaussianCopula", "Hybrid"],
        "test_size": ["Original", "Anonymous", "CTGAN", "TVAE", "GaussianCopula", "Hybrid"],
    }

    def get_anon_others(xp):
        anon_params = ["k_anonymity", "l_diversity", "suppression_limit"]
        if xp in anon_params:
            return [p for p in anon_params if p != xp]
        return []

    os.makedirs(output_dir, exist_ok=True)

    for xparam in xparams:
        if xparam not in df.columns:
            print(f"❌ Skipping xparam '{xparam}' – not in DataFrame.")
            continue

        xparam_dir = os.path.join(output_dir, xparam)
        os.makedirs(xparam_dir, exist_ok=True)

        # Filter by relevant datasets
        allowed_ds = PARAM_DATASET_MAP.get(xparam, [])
        df_xparam = df[df["Dataset"].isin(allowed_ds)]
        if df_xparam.empty:
            print(f"⚠️ Skipping xparam='{xparam}' – no rows after dataset filter.")
            continue

        other_anon = get_anon_others(xparam)
        if other_anon:
            combos = list(product(*[sorted(df_xparam[p].dropna().unique()) for p in other_anon]))
        else:
            combos = [()]

        for metric in metrics:
            metric_dir = os.path.join(xparam_dir, metric)
            os.makedirs(metric_dir, exist_ok=True)

            for combo in combos:
                df_filtered = df_xparam.copy()
                combo_suffix_parts = []

                for p, v in zip(other_anon, combo):
                    df_filtered = df_filtered[df_filtered[p] == v]
                    combo_suffix_parts.append(f"{p}={val_to_str(v)}")

                df_filtered = df_filtered.dropna(subset=[xparam, metric])
                if df_filtered.empty:
                    continue

                # --- NEW: Loop over models separately ---
                for model_name in df_filtered["Model"].dropna().unique():
                    model_sub = df_filtered[df_filtered["Model"] == model_name]
                    if model_sub.empty:
                        continue

                    # Create subfolder for each model
                    model_dir = os.path.join(metric_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)

                    plt.figure(figsize=(8, 5))
                    has_data = False

                    for ds_type in model_sub["Dataset"].unique():
                        ds_sub = model_sub[model_sub["Dataset"] == ds_type].sort_values(by=xparam)
                        if not ds_sub.empty:
                            has_data = True
                            plt.plot(ds_sub[xparam], ds_sub[metric], label=ds_type, marker="o")

                    if not has_data:
                        plt.close()
                        continue

                    plt.xlabel(xparam)
                    plt.ylabel(metric)
                    if combo_suffix_parts:
                        suffix_str = "_".join(combo_suffix_parts)
                        suffix_str_title = "; ".join(combo_suffix_parts)
                    else:
                        suffix_str = "default"
                        suffix_str_title = "default"

                    plt.title(f"{metric} vs {xparam} for {model_name}\n({suffix_str_title})")
                    plt.legend()
                    plt.grid(True)

                    # Save plot and data
                    png_name = f"{metric}_vs_{xparam}_{suffix_str}.png"
                    png_path = os.path.join(model_dir, png_name)

                    csv_name = f"{metric}_vs_{xparam}_{suffix_str}_data.csv"
                    csv_path = os.path.join(model_dir, csv_name)

                    model_sub.to_csv(csv_path, index=False)

                    plt.tight_layout()
                    plt.savefig(png_path)
                    plt.close()

    print(f"📊 Plots + CSV data saved under: {output_dir}")
