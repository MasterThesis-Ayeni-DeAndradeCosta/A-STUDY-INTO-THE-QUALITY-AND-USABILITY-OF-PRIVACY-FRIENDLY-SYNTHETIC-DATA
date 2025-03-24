import os
import yaml
import pandas as pd
import plotly.express as px

def analyze_batch_results(master_batch_folder_path):
    # Load parameter info
    variation_info_path = os.path.join(master_batch_folder_path, "variation_info.yaml")
    with open(variation_info_path, 'r') as f:
        variation_info = yaml.safe_load(f)
    
    param_names = variation_info["varied_parameters"]  # This is a list

    all_dfs = []

    # Gather all result CSVs + inject param values into each df
    for run_folder in os.listdir(master_batch_folder_path):
        run_path = os.path.join(master_batch_folder_path, run_folder)
        if not os.path.isdir(run_path):
            continue

        result_csv = os.path.join(run_path, "model_performance.csv")
        if os.path.exists(result_csv):
            df = pd.read_csv(result_csv)
            import re
            for name in param_names:
                match = re.search(rf"{name}(\d+)", run_folder)
                if match:
                    param_value = int(match.group(1))
                    df[name] = param_value
            all_dfs.append(df)

    if not all_dfs:
        print("❌ No result CSVs found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined results CSV
    analysis_dir = os.path.join(master_batch_folder_path, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    combined_csv_path = os.path.join(analysis_dir, "combined_results.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✅ Combined results saved to: {combined_csv_path}")

    # Detect metrics dynamically
    non_metrics = {"Model", "Dataset", *param_names}
    metric_columns = [col for col in combined_df.columns if col not in non_metrics and pd.api.types.is_numeric_dtype(combined_df[col])]
    print(f"📊 Metrics detected: {metric_columns}")

    # Generate interactive Plotly plots
    for param_name in param_names:
        for metric in metric_columns:
            fig = px.line(
                combined_df, 
                x=param_name, 
                y=metric,
                color="Dataset", 
                line_dash="Model", 
                markers=True,
                title=f"{metric} vs {param_name.capitalize()} (All Datasets + Models)"
            )
            fig.update_layout(template="plotly_white")
            plot_path = os.path.join(analysis_dir, f"{metric.lower()}_vs_{param_name}.html")
            fig.write_html(plot_path)
            print(f"✅ Interactive plot saved: {plot_path}")

    # Export Excel: One sheet per metric, grouped by all param_names
    excel_path = os.path.join(analysis_dir, "combined_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined_df.to_excel(writer, sheet_name="All_Results", index=False)

        for metric in metric_columns:
            pivot_df = combined_df.pivot_table(index=param_names, columns=["Dataset", "Model"], values=metric)
            pivot_df.to_excel(writer, sheet_name=metric[:31], index=True)  # Excel sheet limit 31 chars

        # Summary sheet: Max value per metric
        summary = combined_df.groupby(["Dataset", "Model"])[metric_columns].max().reset_index()
        summary.to_excel(writer, sheet_name="Metric_Summary_Max", index=False)

    print(f"📈 Excel file saved: {excel_path}")
    print("\n🎯 Analysis complete for all metrics.")
