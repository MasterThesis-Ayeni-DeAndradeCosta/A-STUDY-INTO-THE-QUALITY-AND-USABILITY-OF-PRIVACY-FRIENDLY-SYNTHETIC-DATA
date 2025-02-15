import pandas as pd

def encode_categorical_features(original_data, target_column):
    """
    Encodes categorical features using One-Hot Encoding.

    Parameters:
    - original_data (DataFrame): The dataset.
    - target_column (str): The column to exclude from encoding.

    Returns:
    - encoded_data (DataFrame): Transformed dataset with categorical encoding.
    """
    categorical_cols = [col for col in original_data.columns if original_data[col].dtype == "object" and col != target_column]

    if categorical_cols:
        encoded_data = pd.get_dummies(original_data, columns=categorical_cols, drop_first=True)
        print(f"Categorical features encoded: {categorical_cols}")
    else:
        encoded_data = original_data.copy()

    return encoded_data
