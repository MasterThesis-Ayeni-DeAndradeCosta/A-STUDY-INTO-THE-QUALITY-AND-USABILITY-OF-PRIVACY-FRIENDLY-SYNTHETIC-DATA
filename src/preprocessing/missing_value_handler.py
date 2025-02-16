import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(original_data, strategy="drop"):
    """
    Handles missing values in the dataset.

    Parameters:
    - original_data (DataFrame): Input dataset.
     strategy (str or None): "drop" to remove, "mean", "median", or "mode" to impute.
                               If None, returns the original dataset.

    Returns:
    - cleaned_data (DataFrame): Dataset after handling missing values.
    
    """
    if strategy is None:
        return original_data
    
    # Count how many rows we had initially
    original_count = len(original_data)

    if strategy == "drop":
        cleaned_data = original_data.dropna()
    else:
        imputer = SimpleImputer(strategy=strategy)
        cleaned_data = pd.DataFrame(imputer.fit_transform(original_data), columns=original_data.columns)

    # Calculate how many rows were dropped
    dropped_rows = original_count - len(cleaned_data)
    
    # Print a message to see how many were dropped
    print(f"Dropped {dropped_rows} rows due to missing values")

    return cleaned_data
