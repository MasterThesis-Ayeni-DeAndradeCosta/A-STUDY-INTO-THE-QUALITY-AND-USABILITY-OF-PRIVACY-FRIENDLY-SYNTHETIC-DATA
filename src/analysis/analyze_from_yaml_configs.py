import os
import yaml
import pandas as pd
import plotly.express as px
from itertools import product

def val_to_str(val):
    return str(val).replace(".", "p")

def extract_param_value(config, param):
    """Traverse nested dictionaries using dot notation."""
    keys = param.split(".")
    val = config
    try:
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return None

def analyze_batch_results_from_configs(batch_folder):
    variation_info_path = os.path.join(batch_folder, "variation_info.yaml")
    if not os.path.exists(variation_info_path):
        print("❌ No variation_info.yaml found. Skipping analysis.")
        return

    with open(variation_info_path, "r") as f:
        variation_info = yaml.safe_load(f)

    param_names = variation_info.get("varied_parameters", [])
    all_dfs = []

    for run_folder in os.listdir(batch_folder):
        run_path = os.path.join(batch_folder, run_folder)
        if not os.path.isdir(run_path):
            continue

        result_csv = os.path.join(run_path, "model_performance.csv")
        config_path = os.path.join("configs", "generated_configs", "loan", run_folder + ".yaml")

        if not os.path.exists(result_csv) or not os.path.exists(config_path):
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        df = pd.read_csv(result_csv)
        for param in param_names:
            value = extract_param_value(config, param)
            df[param] = value
        all_dfs.append(df)

    if not all_dfs:
        print("❌ No result CSVs found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Folder for analysis
    analysis_dir = os.path.join(batch_folder, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    combined_df.to_csv(os.path.join(analysis_dir, "combined_results.csv"), index=False)

    # Detect metrics
    non_metrics = {"Model", "Dataset", *param_names}
    metric_columns = [c for c in combined_df.columns if c not in non_metrics and pd.api.types.is_numeric_dtype(combined_df[c])]

    # Param-to-dataset applicability
    PARAM_DATASET_MAP = {
        "epochs": ["CTGAN", "TVAE", "GaussianCopula"],
        "k_anonymity": ["Anonymous"]
    }

    # Plotting
    for metric in metric_columns:
        metric_dir = os.path.join(analysis_dir, metric.lower())
        os.makedirs(metric_dir, exist_ok=True)

        for sweep_param in param_names:
            sweep_values = sorted(combined_df[sweep_param].dropna().unique())
            fixed_params = [p for p in param_names if p != sweep_param]
            sweep_dir = os.path.join(metric_dir, f"{metric.lower()}_vs_{sweep_param}")
            os.makedirs(sweep_dir, exist_ok=True)

            fixed_combos = list(product(*[sorted(combined_df[p].dropna().unique()) for p in fixed_params])) or [()]

            for combo in fixed_combos:
                df_filtered = combined_df.copy()
                suffix_parts = []

                for p, v in zip(fixed_params, combo):
                    df_filtered = df_filtered[df_filtered[p] == v]
                    suffix_parts.append(f"{p}{val_to_str(v)}")

                # Safely restrict by dataset only if applicable rows exist
                if sweep_param in PARAM_DATASET_MAP:
                    valid_datasets = PARAM_DATASET_MAP[sweep_param]
                    if df_filtered["Dataset"].isin(valid_datasets).any():
                        df_filtered = df_filtered[df_filtered["Dataset"].isin(valid_datasets)]

                if df_filtered.empty:
                    continue

                df_filtered["Legend"] = df_filtered["Model"] + " | " + df_filtered["Dataset"]

                fig = px.line(
                    df_filtered,
                    x=sweep_param,
                    y=metric,
                    color="Legend",
                    markers=True,
                    hover_data=param_names,
                    title=f"{metric} vs {sweep_param} | {' '.join(suffix_parts)}"
                )
                fig.update_layout(template="plotly_white")

                suffix = "_".join(suffix_parts) if suffix_parts else "all"
                fig.write_html(os.path.join(sweep_dir, f"{suffix}.html"))

    # Excel export
    excel_path = os.path.join(analysis_dir, "combined_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="All_Results", index=False)

        for metric in metric_columns:
            summary = combined_df.groupby(param_names + ["Dataset", "Model"])[metric].mean().reset_index()
            top = summary.loc[summary.groupby(["Dataset", "Model"])[metric].idxmax()]
            top.to_excel(writer, sheet_name=f"Top_{metric[:28]}", index=False)

        pivot_summary = combined_df.groupby(["Dataset", "Model"])[metric_columns].max().reset_index()
        pivot_summary.to_excel(writer, sheet_name="Metric_Summary_Max", index=False)

    print(f"🎉 YAML-based analysis complete. Results saved to: {analysis_dir}")
