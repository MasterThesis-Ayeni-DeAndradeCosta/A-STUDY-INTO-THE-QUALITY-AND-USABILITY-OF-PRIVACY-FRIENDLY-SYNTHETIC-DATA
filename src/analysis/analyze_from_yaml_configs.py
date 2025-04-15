import os
import yaml
import pandas as pd
from collections import defaultdict


def val_to_str(val):
    return str(val).replace(".", "p")


def extract_config_metadata(config_path, dataset_type):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    meta = {}
    meta["test_size"] = config.get("dataset", {}).get("test_size")

    if dataset_type == "Anonymous":
        anon = config.get("anonymization", {})
        meta["k_anonymity"] = anon.get("models", {}).get("k_anonymity")
        meta["l_diversity"] = anon.get("models", {}).get("l_diversity", {}).get("value")
        meta["suppression_limit"] = anon.get("suppression_limit")
        meta["epochs"] = None
        meta["custom_generated_rows"] = None

    elif dataset_type in ["CTGAN", "TVAE", "GaussianCopula", "Hybrid"]:
        meta["k_anonymity"] = meta["l_diversity"] = meta["suppression_limit"] = None
        for synth_name, synth_conf in config.get("synthesis", {}).get("synthesizers", {}).items():
            if synth_conf.get("enabled"):
                meta["epochs"] = synth_conf.get("params", {}).get("epochs")
                meta["custom_generated_rows"] = synth_conf.get("custom_generated_rows")
                break
        else:
            meta["epochs"] = meta["custom_generated_rows"] = None
    else:
        meta["k_anonymity"] = meta["l_diversity"] = meta["suppression_limit"] = None
        meta["epochs"] = meta["custom_generated_rows"] = None

    return meta


def generate_combined_df(batch_folder):
    all_dfs = []

    for run_folder in os.listdir(batch_folder):
        run_path = os.path.join(batch_folder, run_folder)
        if not os.path.isdir(run_path):
            continue

        result_csv = os.path.join(run_path, "model_performance.csv")
        config_path_guess = os.path.join("configs", "generated_configs")
        config_path = None

        for root, _, files in os.walk(config_path_guess):
            for f in files:
                if f == f"{run_folder}.yaml":
                    config_path = os.path.join(root, f)
                    break
            if config_path:
                break

        if not os.path.exists(result_csv) or not config_path:
            continue

        df = pd.read_csv(result_csv)

        for dataset_type in df["Dataset"].unique():
            meta = extract_config_metadata(config_path, dataset_type)
            for key, value in meta.items():
                df.loc[df["Dataset"] == dataset_type, key] = value

        all_dfs.append(df)

    if not all_dfs:
        print("❌ No valid CSVs found.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def generate_best_settings_sheets(excel_path, df):
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        config_types = {
            "Anonymous": "anon_best",
            "Original": "original_best",
            "Hybrid": "hybrid_best",
        }

        for dataset_type, sheet_prefix in config_types.items():
            subset = df[df["Dataset"] == dataset_type]

            if subset.empty:
                print(f"⚠️ No rows found for dataset '{dataset_type}'")
                continue

            for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]:
                metric_vals = subset[metric]
                if metric_vals.notna().any():
                    idx = metric_vals.idxmax()
                    best_row = subset.loc[[idx]]
                    best_row.to_excel(writer, sheet_name=f"{sheet_prefix}_{metric}", index=False)
                else:
                    print(f"⚠️ Skipping {sheet_prefix}_{metric}: all values are NaN")

        def get_best_group(df_sub):
            metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
            df_sub = df_sub.copy()
            df_sub["mean_score"] = df_sub[metric_cols].mean(axis=1, skipna=True)
            best_idx = df_sub.groupby("Model")["mean_score"].idxmax()
            return df_sub.loc[best_idx]

        anon_best = get_best_group(df[df["Dataset"] == "Anonymous"])
        anon_best.to_excel(writer, sheet_name="best_anonymization_settings", index=False)

        synth_best = get_best_group(df[df["Dataset"].isin(["CTGAN", "TVAE", "GaussianCopula", "Hybrid"])])
        synth_best.to_excel(writer, sheet_name="best_synthetic_settings", index=False)


def analyze_batch_results_from_configs(batch_folder):
    combined_df = generate_combined_df(batch_folder)
    if combined_df is None:
        return

    analysis_dir = os.path.join(batch_folder, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    excel_path = os.path.join(analysis_dir, "combined_results.xlsx")
    combined_df.to_excel(excel_path, index=False, sheet_name="combined_results")
    generate_best_settings_sheets(excel_path, combined_df)

    print(f"✅ Analysis complete: {excel_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", required=True, help="Path to batch output folder")
    args = parser.parse_args()
    analyze_batch_results_from_configs(args.batch_dir)
