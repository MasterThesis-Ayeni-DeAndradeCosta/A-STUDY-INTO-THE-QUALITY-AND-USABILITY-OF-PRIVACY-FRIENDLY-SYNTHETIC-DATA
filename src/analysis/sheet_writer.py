import pandas as pd

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
