import pandas as pd
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.impute import SimpleImputer


def handle_missing_values(original_data: pd.DataFrame, strategy: str | None = "drop") -> pd.DataFrame:
    """Clean missing values according to the chosen *strategy*.

    Parameters
    ----------
    original_data : pd.DataFrame
        Input dataset.
    strategy : {"drop", "mean", "median", "mode", None}, default "drop"
        How to treat missing values.

        * ``None``: return *original_data* untouched.
        * ``"drop"``: drop **rows** containing *any* missing value.
        * ``"mean"`` / ``"median"``: impute **numeric** columns with the
          respective statistic.
        * ``"mode"``: impute **all** columns with the most frequent value on a
          per‑column basis (categorical & numeric).

    Returns
    -------
    pd.DataFrame
        A **copy** of the data with missing values handled.
    """

    # ------------------------------------------------------------------
    # 0. Early exit – nothing to do
    # ------------------------------------------------------------------
    if strategy is None:
        return original_data.copy()

    df = original_data.copy()

    # Keep a reference for reporting purposes only
    n_rows_before = len(df)

    # ------------------------------------------------------------------
    # 1. Drop rows containing *any* missing value
    # ------------------------------------------------------------------
    if strategy == "drop":
        cleaned = df.dropna()
        n_dropped = n_rows_before - len(cleaned)
        print(f"Dropped {n_dropped} rows that contained at least one missing value.")
        return cleaned

    # ------------------------------------------------------------------
    # 2. Mean / Median imputation **numeric columns only**
    # ------------------------------------------------------------------
    if strategy in {"mean", "median"}:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if numeric_cols.size:
            imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

            # Restore integer dtypes where applicable (ARX requires ints)
            for col in numeric_cols:
                if pd.api.types.is_integer_dtype(original_data[col]):
                    # Use nullable Int64 to keep NA compatibility
                    df[col] = df[col].round().astype("Int64")
            print(f"Imputed {numeric_cols.size} numeric column(s) using {strategy}.")
        else:
            print("No numeric columns found – nothing to impute with mean/median.")
        return df

    # ------------------------------------------------------------------
    # 3. Mode imputation (works for *all* dtypes)
    # ------------------------------------------------------------------
    if strategy == "mode":
        for col in df.columns:
            if df[col].isna().any():
                mode_vals = df[col].mode(dropna=True)
                if not mode_vals.empty:
                    df[col].fillna(mode_vals.iloc[0], inplace=True)
        print("Filled missing values in each column with its respective mode.")
        return df

    # ------------------------------------------------------------------
    # 4. Unknown strategy – raise early so caller knows
    # ------------------------------------------------------------------
    raise ValueError(
        "Unsupported missing value strategy: '{strategy}'. "
        "Choose from 'drop', 'mean', 'median', 'mode', or None.".format(strategy=strategy)
    )


# def handle_missing_values(original_data, strategy="drop"):
#     """
#     Handles missing values in the dataset.

#     Parameters:
#     - original_data (DataFrame): Input dataset.
#     - strategy (str or None): "drop" to remove, "mean", "median", or "mode" to impute.

#     Returns:
#     - cleaned_data (DataFrame): Dataset after handling missing values.
#     """
#     if strategy is None:
#         return original_data

#     original_count = len(original_data)

#     df = original_data.copy()

#     if strategy == "drop":
#         cleaned_data = df.dropna()
#     elif strategy in ["mean", "median"]:
#         numeric_cols = df.select_dtypes(include=["number"]).columns
#         imputer = SimpleImputer(strategy=strategy)
#         df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
#         # Restore integer type where possible (important for ARX)
#         for col in numeric_cols:
#             if pd.api.types.is_integer_dtype(original_data[col]):
#                 df[col] = df[col].round().astype(int)
#             cleaned_data = df

#     elif strategy == "mode":
#         for col in df.columns:
#             if df[col].isnull().any():
#                 df[col].fillna(df[col].mode()[0], inplace=True)
#         cleaned_data = df
#     else:
#         raise ValueError(f"Unsupported missing value strategy: {strategy}")

#     dropped_rows = original_count - len(cleaned_data)
#     print(f"Dropped {dropped_rows} rows due to missing values")

#     return cleaned_data
