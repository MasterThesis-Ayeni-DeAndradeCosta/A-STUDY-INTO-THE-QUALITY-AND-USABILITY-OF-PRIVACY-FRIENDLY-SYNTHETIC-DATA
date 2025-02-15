import pandas as pd
import category_encoders as ce

def encode_categorical_features(original_data, target_column):
    """
    Encodes categorical features using Binary Encoding and provides detailed print statements.

    Parameters:
    - original_data (DataFrame): The dataset.
    - target_column (str): The column to exclude from encoding.

    Returns:
    - encoded_data (DataFrame): Transformed dataset with binary encoding.
    """
    # Identify categorical columns (excluding target column)
    categorical_cols = [col for col in original_data.columns if original_data[col].dtype == "object" and col != target_column]
    
    print(f"\n🔹 Identified Categorical Columns: {categorical_cols}")
    
    if categorical_cols:
        # Initialize Binary Encoder
        encoder = ce.BinaryEncoder(cols=categorical_cols, drop_invariant=True)

        # Print before transformation
        original_shape = original_data.shape
        print(f"Original Data Shape: {original_shape}")

        # Perform Encoding
        encoded_data = encoder.fit_transform(original_data)

        # Print transformation details
        new_columns = list(set(encoded_data.columns) - set(original_data.columns))
        print(f"✅ Binary Encoding applied. New Features Added: {new_columns}")

        # Print final shape after encoding
        new_shape = encoded_data.shape
        print(f"New Data Shape after Encoding: {new_shape}")

    else:
        print("⚠ No categorical columns found. Returning original data.")
        encoded_data = original_data.copy()

    return encoded_data
