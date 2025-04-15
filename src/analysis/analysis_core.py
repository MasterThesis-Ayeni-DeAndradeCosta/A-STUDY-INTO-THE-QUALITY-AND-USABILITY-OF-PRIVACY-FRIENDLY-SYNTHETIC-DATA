import os
import yaml
import pandas as pd

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
        meta["k_anonymity"] = None
        meta["l_diversity"] = None
        meta["suppression_limit"] = None
        # find whichever synthesizer is enabled
        for synth_name, synth_conf in config.get("synthesis", {}).get("synthesizers", {}).items():
            if synth_conf.get("enabled"):
                meta["epochs"] = synth_conf.get("params", {}).get("epochs")
                meta["custom_generated_rows"] = synth_conf.get("custom_generated_rows")
                break
        else:
            meta["epochs"] = None
            meta["custom_generated_rows"] = None

    else:
        meta["k_anonymity"] = None
        meta["l_diversity"] = None
        meta["suppression_limit"] = None
        meta["epochs"] = None
        meta["custom_generated_rows"] = None

    return meta


def generate_combined_df(batch_folder):
    """
    Collect all run folders inside 'batch_folder', read their model_performance.csv,
    attach config metadata, return a combined DataFrame.
    """
    all_dfs = []
    print(f"🔍 Scanning batch folder: {batch_folder}")

    for run_name in os.listdir(batch_folder):
        run_path = os.path.join(batch_folder, run_name)
        if not os.path.isdir(run_path):
            continue
        csv_path = os.path.join(run_path, "model_performance.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ No model_performance.csv in {run_name}")
            continue

        # Attempt to find config matching run_name.yaml
        config_path = None
        config_guess = os.path.join("configs", "generated_configs")
        for root, dirs, files in os.walk(config_guess):
            if f"{run_name}.yaml" in files:
                config_path = os.path.join(root, f"{run_name}.yaml")
                break
        if not config_path:
            print(f"⚠️ No config for {run_name}")
            continue

        df_part = pd.read_csv(csv_path)
        if df_part.empty or df_part.isna().all().all():
            print(f"⚠️ Empty results for {run_name}")
            continue

        # Attach metadata per dataset
        for ds_type in df_part["Dataset"].unique():
            meta = extract_config_metadata(config_path, ds_type)
            for k,v in meta.items():
                df_part.loc[df_part["Dataset"]==ds_type, k] = v

        all_dfs.append(df_part)

    if not all_dfs:
        print("❌ No valid CSVs found.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Combined {len(all_dfs)} runs.")
    return combined_df


def save_combined_results_to_excel(df, analysis_dir):
    """Write df to 'combined_results.xlsx' in 'analysis_dir'."""
    os.makedirs(analysis_dir, exist_ok=True)
    excel_path = os.path.join(analysis_dir, "combined_results.xlsx")
    df.to_excel(excel_path, index=False, sheet_name="combined_results")
    print(f"📁 Saved combined results to: {excel_path}")
    return excel_path
