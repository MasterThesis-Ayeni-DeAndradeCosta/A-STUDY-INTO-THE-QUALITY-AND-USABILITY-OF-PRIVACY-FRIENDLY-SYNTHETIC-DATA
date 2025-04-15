import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from analysis.analysis_core import generate_combined_df, save_combined_results_to_excel
from analysis.sheet_writer import generate_best_settings_sheets
from analysis.plotting_utils import generate_static_plots



def run_batch_analysis(batch_dir: str):
    print(f"📊 Running batch analysis on: {batch_dir}")

    # 1. Generate combined DataFrame from all run folders
    df = generate_combined_df(batch_dir)
    if df is None:
        print("❌ No valid results found.")
        return

    # 2. Create the output analysis directory
    analysis_dir = os.path.join(batch_dir, "batch_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 3. Save combined DataFrame to Excel
    excel_path = save_combined_results_to_excel(df, analysis_dir)

    # 4. Generate best settings sheets for each type (Anon, Original, Synth)
    generate_best_settings_sheets(excel_path, df)

    # Step 5. Generate plots
    variation_info_path = os.path.join(batch_dir, "variation_info.yaml")
    generate_static_plots(df, variation_info_path, analysis_dir)

    print(f"✅ Analysis complete: {excel_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", required=True, help="Path to batch output folder")
    args = parser.parse_args()

    run_batch_analysis(args.batch_dir)
