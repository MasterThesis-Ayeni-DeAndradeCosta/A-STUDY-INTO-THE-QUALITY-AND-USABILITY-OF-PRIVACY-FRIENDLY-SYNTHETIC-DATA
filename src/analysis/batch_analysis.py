import os
import yaml
import pandas as pd
import plotly.express as px

def analyze_batch_results(master_batch_folder_path):
    # Load parameter info
    variation_info_path = os.path.join(master_batch_folder_path, "variation_info.yaml")
    with open(variation_info_path, 'r') as f:
        variation_info = yaml.safe_load(f)
    param_name = variation_info["varied_parameter"]

    all_dfs = []

    for run_folder in os.listdir(master_batch_folder_path):
        run_path = os.path.join(master_batch_folder_path, run_folder)
        if not os.path.isdir(run_path):
            continue

        result_csv = os.path.join(run_path, "model_performance.csv")
        if os.path.exists(result_csv):
            df = pd.read_csv(result_csv)
            try:
                import re
                match = re.search(rf"{param_name}(\d+)", run_folder)
                if match:
                    param_value = int(match.group(1))
                    df[param_name] = param_value
                    all_dfs.append(df)
            except ValueError:
                continue

    if not all_dfs:
        print("❌ No result CSVs found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Create analysis folder
    analysis_folder = os.path.join(master_batch_folder_path, "analysis")
    os.makedirs(analysis_folder, exist_ok=True)

    # Save combined results CSV
    combined_csv_path = os.path.join(analysis_folder, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✅ Combined results saved to: {combined_csv_path}")

    # Detect metric columns
    non_metrics = {"Model", "Dataset", param_name}
    metric_columns = [col for col in combined_df.columns if col not in non_metrics and pd.api.types.is_numeric_dtype(combined_df[col])]

    print(f"📊 Metrics detected: {metric_columns}")

    # Plot interactive charts for each metric
    for metric in metric_columns:
        fig = px.line(
            combined_df, 
            x=param_name, 
            y=metric, 
            color="Dataset", 
            line_dash="Model", 
            markers=True,
            title=f"{metric} vs {param_name.capitalize()} (Interactive)"
        )
        fig.update_layout(template="plotly_white")
        plot_path = os.path.join(analysis_folder, f"{metric.lower()}_vs_{param_name}.html")
        fig.write_html(plot_path)
        print(f"✅ Interactive plot saved: {plot_path}")

    # Export Excel with raw and summary stats
    excel_path = os.path.join(analysis_folder, "metrics_summary.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        combined_df.to_excel(writer, sheet_name="All_Results", index=False)

        summary = combined_df.groupby(["Dataset", "Model", param_name])[metric_columns].agg(["mean", "std", "max", "min"])
        summary.to_excel(writer, sheet_name="Summary_Stats")

    print(f"📊 Excel summary saved: {excel_path}")
    print("\n🎯 Analysis complete for all metrics.")
