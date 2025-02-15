import pandas as pd
import category_encoders as ce

def encode_categorical_features(original_data, target_column):
    """
    Encodes categorical features using Binary Encoding.

    Parameters:
    - original_data (DataFrame): The dataset.
    - target_column (str): The column to exclude from encoding.

    Returns:
    - encoded_data (DataFrame): Transformed dataset with binary encoding.
    """
    categorical_cols = [col for col in original_data.columns if original_data[col].dtype == "object" and col != target_column]

    if categorical_cols:
        encoder = ce.BinaryEncoder(cols=categorical_cols, drop_invariant=True)
        encoded_data = encoder.fit_transform(original_data)
        print(f"Categorical features encoded using Binary Encoding: {categorical_cols}")
    else:
        encoded_data = original_data.copy()

    return encoded_data
