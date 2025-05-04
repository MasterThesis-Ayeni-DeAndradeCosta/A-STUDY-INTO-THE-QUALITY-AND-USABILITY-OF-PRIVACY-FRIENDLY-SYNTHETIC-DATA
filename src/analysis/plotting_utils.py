import os
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import yaml

def val_to_str(val):
    return str(val).replace(".", "p")

def generate_static_plots(df, variation_info_path, output_dir):
    """
    Generates static PNG plots (and matching CSVs) for every metric / x‑parameter
    combination defined in *variation_info.yaml*.

    - Epochs plots ➔ Keep old behavior.
    - Row Multiplier plots ➔ NEW: Fix `epochs` value during plotting.
    - Hybrid data ➔ Included for anonymization parameters (k, l, suppression_limit).
    """
    if not os.path.exists(variation_info_path):
        print("⚠️ No variation_info.yaml found. Skipping plots.")
        return

    with open(variation_info_path, "r") as f:
        var_info = yaml.safe_load(f)

    xparams = var_info.get("varied_parameters", [])
    metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"] if m in df.columns]

    # Which datasets are relevant for each parameter sweep
    PARAM_DATASET_MAP = {
        "k_anonymity": ["Anonymous", "_HYBRID"],  # allow Anonymous + Hybrid
        "l_diversity": ["Anonymous", "_HYBRID"],
        "suppression_limit": ["Anonymous", "_HYBRID"],
        "epochs": ["CTGAN", "TVAE", "GaussianCopula", "Hybrid"],
        "row_multiplier": ["CTGAN", "TVAE", "GaussianCopula", "Hybrid"],
        "test_size": ["Original", "Anonymous", "CTGAN", "TVAE", "GaussianCopula", "Hybrid"],
    }

    def get_anon_others(xp: str):
        anon_params = ["k_anonymity", "l_diversity", "suppression_limit"]
        return [p for p in anon_params if p != xp] if xp in anon_params else []

    os.makedirs(output_dir, exist_ok=True)

    for xparam in xparams:
        if xparam not in df.columns:
            print(f"❌ Skipping xparam '{xparam}' – not in DataFrame.")
            continue

        xparam_dir = os.path.join(output_dir, xparam)
        os.makedirs(xparam_dir, exist_ok=True)

        allowed_ds = PARAM_DATASET_MAP.get(xparam, [])
        if "_HYBRID" in allowed_ds:
            df_xparam = df[df["Dataset"].apply(lambda x: x == "Anonymous" or x.endswith("_HYBRID"))].copy()
        else:
            df_xparam = df[df["Dataset"].isin(allowed_ds)].copy()

        if df_xparam.empty:
            print(f"⚠️ Skipping xparam='{xparam}' – no rows after dataset filter.")
            continue

        other_anon = get_anon_others(xparam)
        combos = list(product(*[sorted(df_xparam[p].dropna().unique()) for p in other_anon])) or [()]

        for metric in metrics:
            metric_dir = os.path.join(xparam_dir, metric)
            os.makedirs(metric_dir, exist_ok=True)

            for combo in combos:
                df_filtered = df_xparam.copy()
                combo_suffix_parts = []

                abbrev = {"k_anonymity": "k", "l_diversity": "l", "suppression_limit": "suplim"}
                for p, v in zip(other_anon, combo):
                    df_filtered = df_filtered[df_filtered[p] == v]
                    combo_suffix_parts.append(f"{abbrev.get(p, p)}{val_to_str(v)}")

                df_filtered = df_filtered.dropna(subset=[xparam, metric])
                if df_filtered.empty:
                    continue

                # ➔ Collapsing duplicates if needed
                if xparam == "epochs" and "row_multiplier" in df_filtered.columns:
                    df_filtered = (
                        df_filtered
                        .groupby([xparam, "Dataset", "Model"], as_index=False)[metric]
                        .mean()
                    )

                # -------------------
                # DIFFERENT BEHAVIOR FOR ROW_MULTIPLIER
                # -------------------
                if xparam == "row_multiplier":
                    for epochs_val in sorted(df_filtered["epochs"].dropna().unique()):
                        df_epochs_fixed = df_filtered[df_filtered["epochs"] == epochs_val]

                        if df_epochs_fixed.empty:
                            continue

                        for model_name in df_epochs_fixed["Model"].dropna().unique():
                            model_sub = df_epochs_fixed[df_epochs_fixed["Model"] == model_name]
                            if model_sub.empty:
                                continue

                            fixed_epochs_folder = os.path.join(metric_dir, model_name, f"epochs{epochs_val}")
                            os.makedirs(fixed_epochs_folder, exist_ok=True)

                            plt.figure(figsize=(8, 5))
                            for ds_type in model_sub["Dataset"].unique():
                                ds_sub = model_sub[model_sub["Dataset"] == ds_type].sort_values(by=xparam)
                                plt.plot(ds_sub[xparam], ds_sub[metric], label=ds_type, marker="o")

                            plt.xlabel(xparam)
                            plt.ylabel(metric)

                            if combo_suffix_parts:
                                suffix_str = "_".join(combo_suffix_parts)
                                suffix_title = "; ".join(combo_suffix_parts)
                            else:
                                suffix_str = suffix_title = "default"

                            plt.title(f"{metric} vs {xparam} for {model_name}\n(epochs={epochs_val}, {suffix_title})")
                            plt.legend()
                            plt.grid(True)

                            png_name = f"{metric}_vs_{xparam}_{suffix_str}.png"
                            png_path = os.path.join(fixed_epochs_folder, png_name)

                            csv_name = f"{metric}_vs_{xparam}_{suffix_str}_data.csv"
                            csv_path = os.path.join(fixed_epochs_folder, csv_name)

                            model_sub.to_csv(csv_path, index=False)
                            plt.tight_layout()
                            plt.savefig(png_path)
                            plt.close()

                # -------------------
                # NORMAL behavior for other xparams (epochs, suppression, etc.)
                # -------------------
                else:
                    for model_name in df_filtered["Model"].dropna().unique():
                        model_sub = df_filtered[df_filtered["Model"] == model_name]
                        if model_sub.empty:
                            continue

                        model_dir = os.path.join(metric_dir, model_name)
                        os.makedirs(model_dir, exist_ok=True)

                        plt.figure(figsize=(8, 5))
                        for ds_type in model_sub["Dataset"].unique():
                            ds_sub = model_sub[model_sub["Dataset"] == ds_type].sort_values(by=xparam)
                            plt.plot(ds_sub[xparam], ds_sub[metric], label=ds_type, marker="o")

                        plt.xlabel(xparam)
                        plt.ylabel(metric)

                        if combo_suffix_parts:
                            suffix_str = "_".join(combo_suffix_parts)
                            suffix_title = "; ".join(combo_suffix_parts)
                        else:
                            suffix_str = suffix_title = "default"

                        plt.title(f"{metric} vs {xparam} for {model_name}\n({suffix_title})")
                        plt.legend()
                        plt.grid(True)

                        png_name = f"{metric}_vs_{xparam}_{suffix_str}.png"
                        png_path = os.path.join(model_dir, png_name)

                        csv_name = f"{metric}_vs_{xparam}_{suffix_str}_data.csv"
                        csv_path = os.path.join(model_dir, csv_name)

                        model_sub.to_csv(csv_path, index=False)
                        plt.tight_layout()
                        plt.savefig(png_path)
                        plt.close()

    print(f"📊 Plots + CSV data saved under: {output_dir}")
