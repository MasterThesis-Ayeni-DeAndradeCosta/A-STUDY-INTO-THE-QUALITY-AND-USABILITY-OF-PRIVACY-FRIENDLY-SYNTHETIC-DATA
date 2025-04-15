import os
import pandas as pd
from .metadata_utils import extract_config_metadata


def val_to_str(val):
    return str(val).replace(".", "p")


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

    return pd.concat(all_dfs, ignore_index=True)


def save_combined_results_to_excel(df, output_folder):
    excel_path = os.path.join(output_folder, "combined_results.xlsx")
    df.to_excel(excel_path, index=False, sheet_name="combined_results")
    return excel_path
