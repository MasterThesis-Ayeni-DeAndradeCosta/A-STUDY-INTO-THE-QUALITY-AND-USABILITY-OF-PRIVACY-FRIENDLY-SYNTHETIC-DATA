# file: src/analysis/sheet_writer.py

import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import PatternFill

def generate_analysis_sheets(excel_path, df):
    """
    - Leaves existing 'combined_results' alone (written by save_combined_results_to_excel).
    - Creates these new sheets:
        1) Anon_Results  -> pivot or table of all anonymized data
        2) Synth_Results -> pivot or table of all synthetic data
        3) Anon_Best     -> top 3 rows per (Model, Metric) for anonymized
        4) Synth_Best    -> top 3 rows per (Model, Metric) for synthetic
    """
    wb = openpyxl.load_workbook(excel_path)

    # Remove if they already exist
    for sname in ["Anon_Results","Synth_Results","Anon_Best","Synth_Best"]:
        if sname in wb.sheetnames:
            del wb[sname]

    # 1) All anonymized pivot
    ws_anon = wb.create_sheet("Anon_Results")
    _write_anon_results(ws_anon, df)

    # 2) All synthetic pivot
    ws_synth = wb.create_sheet("Synth_Results")
    _write_synth_results(ws_synth, df)

    # 3) Anon_Best: top 3 rows per (Model, Metric)
    ws_anon_best = wb.create_sheet("Anon_Best")
    _write_anon_best(ws_anon_best, df)

    # 4) Synth_Best: top 3 rows per (Model, Metric)
    ws_synth_best = wb.create_sheet("Synth_Best")
    _write_synth_best(ws_synth_best, df)

    # Auto-size columns in each newly created sheet
    for sname in ["Anon_Results","Synth_Results","Anon_Best","Synth_Best"]:
        _autosize_worksheet(wb[sname])

    wb.save(excel_path)
    print(f"✅ Wrote 'Anon_Results','Synth_Results','Anon_Best','Synth_Best' to {excel_path}")

# -------------------------------------------------------------------
# 1) Full pivot for anonymized data
def _write_anon_results(ws, df):
    """
    For Dataset='Anonymous'.
    Group by (k_anonymity, l_diversity, suppression_limit, Model),
    then average [Accuracy, F1, etc.].
    """
    sub = df[df["Dataset"]=="Anonymous"].copy()
    if sub.empty:
        ws["A1"] = "No Anonymous rows found."
        return

    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    metrics = [m for m in metrics if m in sub.columns]
    if not metrics:
        ws["A1"] = "No metrics found for Anonymous data."
        return

    pivot_anon = (
        sub.groupby(["k_anonymity","l_diversity","suppression_limit","Model"])[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    _write_df_to_sheet(ws, pivot_anon)

# -------------------------------------------------------------------
# 2) Full pivot for synthetic data
def _write_synth_results(ws, df):
    """
    For Dataset in [CTGAN,TVAE,GaussianCopula,Hybrid].
    Group by (Dataset, epochs, custom_generated_rows, Model),
    then average metrics.
    """
    sub = df[df["Dataset"].isin(["CTGAN","TVAE","GaussianCopula","Hybrid"])].copy()
    if sub.empty:
        ws["A1"] = "No Synthetic rows found."
        return

    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    metrics = [m for m in metrics if m in sub.columns]
    if not metrics:
        ws["A1"] = "No metrics columns in synth data."
        return

    pivot_synth = (
        sub.groupby(["Dataset","epochs","custom_generated_rows","Model"])[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    _write_df_to_sheet(ws, pivot_synth)

# -------------------------------------------------------------------
# 3) Best 3 anonymized rows per (Model, Metric)
def _write_anon_best(ws, df):
    """
    For anonymized data only.
    For each model, for each metric, pick the top 3 rows.
    """
    sub = df[df["Dataset"]=="Anonymous"].copy()
    if sub.empty:
        ws["A1"] = "No Anonymous data found."
        return

    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    metrics = [m for m in metrics if m in sub.columns]
    if not metrics:
        ws["A1"] = "No metric columns for Anonymous data."
        return

    best_rows = []
    for model in sub["Model"].unique():
        model_df = sub[sub["Model"]==model]
        for metric in metrics:
            # Filter out rows missing that metric
            valid_df = model_df[model_df[metric].notna()]
            if valid_df.empty:
                continue
            # Sort descending by metric, pick top 3
            top_3 = valid_df.nlargest(3, metric)
            for rank, (idx, row_data) in enumerate(top_3.iterrows(), start=1):
                row_data = row_data.copy()
                row_data["WhichMetric"] = metric
                row_data["Rank"] = rank
                row_data["MetricValue"] = row_data[metric]
                best_rows.append(row_data)

    if not best_rows:
        ws["A1"] = "No best rows found for Anonymous."
        return

    best_df = pd.DataFrame(best_rows)
    # reorder columns
    keep_cols = ["Model","WhichMetric","Rank","MetricValue",
                 "k_anonymity","l_diversity","suppression_limit",
                 "Accuracy","Precision","Recall","F1","AUC-ROC"]
    # keep only columns that exist
    keep_cols = [c for c in keep_cols if c in best_df.columns]
    best_df = best_df[keep_cols]

    # optionally sort best_df by (WhichMetric, Rank)
    best_df = best_df.sort_values(by=["WhichMetric","Rank"], ascending=True)

    _write_df_to_sheet(ws, best_df)

# -------------------------------------------------------------------
# 4) Best 3 synthetic rows per (Model, Metric)
def _write_synth_best(ws, df):
    """
    For synthetic data only.
    For each model, for each metric, pick the top 3 rows.
    """
    sub = df[df["Dataset"].isin(["CTGAN","TVAE","GaussianCopula","Hybrid"])]
    if sub.empty:
        ws["A1"] = "No Synthetic data found."
        return

    metrics = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
    metrics = [m for m in metrics if m in sub.columns]
    if not metrics:
        ws["A1"] = "No metric columns for Synthetic data."
        return

    best_rows = []
    for model in sub["Model"].unique():
        model_df = sub[sub["Model"]==model]
        for metric in metrics:
            valid_df = model_df[model_df[metric].notna()]
            if valid_df.empty:
                continue
            top_3 = valid_df.nlargest(3, metric)
            for rank, (idx, row_data) in enumerate(top_3.iterrows(), start=1):
                row_data = row_data.copy()
                row_data["WhichMetric"] = metric
                row_data["Rank"] = rank
                row_data["MetricValue"] = row_data[metric]
                best_rows.append(row_data)

    if not best_rows:
        ws["A1"] = "No best synthetic rows found."
        return

    best_df = pd.DataFrame(best_rows)
    # reorder columns
    keep_cols = ["Dataset","Model","WhichMetric","Rank","MetricValue",
                 "epochs","custom_generated_rows","Accuracy","Precision","Recall","F1","AUC-ROC"]
    keep_cols = [c for c in keep_cols if c in best_df.columns]
    best_df = best_df[keep_cols]

    best_df = best_df.sort_values(by=["WhichMetric","Rank"], ascending=True)

    _write_df_to_sheet(ws, best_df)


# -------------------------------------------------------------------
# HELPER: writes a DataFrame to the sheet
def _write_df_to_sheet(ws, df, start_row=1, start_col=1):
    # header
    for c_idx, col_name in enumerate(df.columns, start_col):
        ws.cell(row=start_row, column=c_idx, value=col_name)
    # data
    for r_idx, row_data in enumerate(df.itertuples(index=False), start_row+1):
        for c_idx, val in enumerate(row_data, start_col):
            ws.cell(row=r_idx, column=c_idx, value=val)

def _autosize_worksheet(ws):
    from openpyxl.utils import get_column_letter
    for col in range(1, ws.max_column+1):
        max_length = 0
        col_letter = get_column_letter(col)
        for row in range(1, ws.max_row+1):
            val = ws.cell(row=row, column=col).value
            if val is not None:
                length = len(str(val))
                if length > max_length:
                    max_length = length
        ws.column_dimensions[col_letter].width = max_length + 2
