import os
import pandas as pd
import plotly.express as px

def run_analysis(run_folder_path, analysis_config):
    # Check if analysis is enabled
    if not analysis_config.get("enable_analysis", True):
        print("⚠️ Singular analysis skipped (disabled in config).")
        return
    
    result_csv = os.path.join(run_folder_path, "model_performance.csv")
    if not os.path.exists(result_csv):
        print(f"❌ model_performance.csv not found in: {run_folder_path}")
        return

    df = pd.read_csv(result_csv)

    # Detect metric columns
    non_metrics = {"Model", "Dataset"}
    metric_columns = [col for col in df.columns if col not in non_metrics and pd.api.types.is_numeric_dtype(df[col])]

    print(f"📊 Metrics detected: {metric_columns}")

    # Create analysis subfolder
    analysis_folder = os.path.join(run_folder_path, "analysis")
    os.makedirs(analysis_folder, exist_ok=True)

    # Save raw results
    raw_copy_path = os.path.join(analysis_folder, "model_performance_copy.csv")
    df.to_csv(raw_copy_path, index=False)
    print(f"✅ Saved raw results: {raw_copy_path}")

    # Plot interactive bar charts
    # Generate interactive plots if enabled
    if analysis_config.get("generate_interactive_plots", True):
        for metric in metric_columns:
            fig = px.bar(
                df,
                x="Model",
                y=metric,
                color="Dataset",
                barmode="group",
                title=f"{metric} Comparison Across Datasets and Models",
                text_auto=".2f"
            )
            fig.update_layout(template="plotly_white")
            plot_path = os.path.join(analysis_folder, f"{metric.lower()}_comparison.html")
            fig.write_html(plot_path)
            print(f"✅ Saved interactive plot: {plot_path}")

    # Generate Excel summary if enabled
    if analysis_config.get("generate_excel_summary", True):
        excel_path = os.path.join(analysis_folder, "metrics_summary.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name="Raw_Results", index=False)
            summary = df.groupby(["Dataset", "Model"])[metric_columns].agg(["mean", "std", "max", "min"])
            summary.to_excel(writer, sheet_name="Summary_Stats")
        print(f"📈 Excel summary saved: {excel_path}")

    print(f"📈 Excel summary saved: {excel_path}")
    print("🎯 Singular run analysis complete.")
