import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(original_data, strategy="drop"):
    """
    Handles missing values in the dataset.

    Parameters:
    - original_data (DataFrame): Input dataset.
    - strategy (str or None): "drop" to remove, "mean", "median", or "mode" to impute.

    Returns:
    - cleaned_data (DataFrame): Dataset after handling missing values.
    """
    if strategy is None:
        return original_data

    original_count = len(original_data)

    df = original_data.copy()

    if strategy == "drop":
        cleaned_data = df.dropna()
    elif strategy in ["mean", "median"]:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        cleaned_data = df
    elif strategy == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        cleaned_data = df
    else:
        raise ValueError(f"Unsupported missing value strategy: {strategy}")

    dropped_rows = original_count - len(cleaned_data)
    print(f"Dropped {dropped_rows} rows due to missing values")

    return cleaned_data
