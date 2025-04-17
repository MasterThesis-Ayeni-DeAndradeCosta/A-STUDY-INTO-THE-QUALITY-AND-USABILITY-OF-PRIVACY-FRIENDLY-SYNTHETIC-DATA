import pandas as pd
import category_encoders as ce


def encode_categorical_features_train_test(original_data, target_column, test_data):
    """
    Encodes categorical features using Binary Encoding and provides detailed print statements.

    Parameters:
    - original_data (DataFrame): The training dataset.
    - target_column (str): The column to exclude from encoding.
    - test_data (DataFrame): The test dataset to transform using the same encoder.

    Returns:
    - cleaned_train (DataFrame): Transformed training dataset with binary encoding.
    - cleaned_test (DataFrame): Transformed test dataset.
    - encoder (BinaryEncoder): The fitted encoder instance.
    - encoding_map (dict): Mapping from original column to newly created encoded columns.
    """
    categorical_cols = [col for col in original_data.columns if original_data[col].dtype == "object" and col != target_column]
    encoding_map = {}
    cleaned_test = None

    print(f"\n🔹 Identified Categorical Columns: {categorical_cols}")
    
    if categorical_cols:
        encoder = ce.BinaryEncoder(cols=categorical_cols, drop_invariant=True)

        original_shape = original_data.shape
        print(f"📐 Original Train Data Shape: {original_shape}")

        # Encode training data
        encoded_array = encoder.fit_transform(original_data)
        cleaned_train = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

        print(f"✅ Training Data Encoding applied.")
        print(f"📐 Encoded Train Data Shape: {cleaned_train.shape}")
        print(f" Number of Encoded Columns: {len(cleaned_train.columns)}")

        # Encode test data
        print(f" Test Data Shape Before Encoding: {test_data.shape}")
        test_encoded_array = encoder.transform(test_data)
        cleaned_test = pd.DataFrame(test_encoded_array, columns=encoder.get_feature_names_out())
        print(f"✅ Test Data Encoding applied.")
        print(f"📐 Encoded Test Data Shape: {cleaned_test.shape}")

        # Check for column mismatch
        if set(cleaned_train.columns) != set(cleaned_test.columns):
            print("🚨 WARNING: Train and test encoded columns do not match!")
            print(f"    Columns in train but not in test: {set(cleaned_train.columns) - set(cleaned_test.columns)}")
            print(f"    Columns in test but not in train: {set(cleaned_test.columns) - set(cleaned_train.columns)}")
        else:
            print("  Train and test encoded columns are aligned.")

        # Compute the mapping: which new columns came from which original column
        all_encoded_cols = list(cleaned_train.columns)
        for col in categorical_cols:
            mapped = [c for c in all_encoded_cols if c.startswith(col + "_")]
            if mapped:
                encoding_map[col] = mapped

        new_columns = list(set(cleaned_train.columns) - set(original_data.columns))
        print(f" New Features Added by Encoding: {new_columns}")

    else:
        print("⚠ No categorical columns found. Returning original data.")
        cleaned_train = original_data.copy()
        cleaned_test = test_data.copy()
        encoder = None

    return cleaned_train, cleaned_test, encoder, encoding_map


def encode_categorical_features(original_data, target_column, encoder=None):
    """
    Encodes categorical features using Binary Encoding and provides detailed print statements.

    Parameters:
    - original_data (DataFrame): The dataset.
    - target_column (str): The column to exclude from encoding.
    - encoder (BinaryEncoder, optional): If provided, reuse for transformation only.

    Returns:
    - encoded_data (DataFrame): Transformed dataset with binary encoding.
    - encoding_map (dict): Mapping from original column to newly created encoded columns.
    """
    categorical_cols = [col for col in original_data.columns if original_data[col].dtype == "object" and col != target_column]
    encoding_map = {}

    print(f"\n🔹 Identified Categorical Columns: {categorical_cols}")
    
    if categorical_cols:
        if encoder is not None:
            print("✅ Using provided encoder to transform data.")
            encoded_array = encoder.transform(original_data)
            encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
            return encoded_data, encoding_map  # Skip mapping, since encoder was reused

        # Default behavior — fit new encoder
        encoder = ce.BinaryEncoder(cols=categorical_cols, drop_invariant=True)
        original_shape = original_data.shape
        print(f"Original Data Shape: {original_shape}")

        encoded_array = encoder.fit_transform(original_data)
        encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

        # Compute the mapping: which new columns came from which original column
        all_encoded_cols = list(encoded_data.columns)
        for col in categorical_cols:
            mapped = [c for c in all_encoded_cols if c.startswith(col + "_")]
            if mapped:
                encoding_map[col] = mapped

        new_columns = list(set(encoded_data.columns) - set(original_data.columns))
        print(f"✅ Binary Encoding applied. New Features Added: {new_columns}")
        print(f"New Data Shape after Encoding: {encoded_data.shape}")

    else:
        print("⚠ No categorical columns found. Returning original data.")
        encoded_data = original_data.copy()

    return encoded_data, encoding_map




# def encode_categorical_features(original_data, target_column, encoder=None):
#     """
#     Encodes categorical features using Binary Encoding and provides detailed print statements.

#     Parameters:
#     - original_data (DataFrame): The dataset.
#     - target_column (str): The column to exclude from encoding.

#     Returns:
#     - encoded_data (DataFrame): Transformed dataset with binary encoding.
#     - encoding_map (dict): Mapping from original column to newly created encoded columns.
#     """
#     categorical_cols = [col for col in original_data.columns if original_data[col].dtype == "object" and col != target_column]
#     encoding_map = {}

#     print(f"\n🔹 Identified Categorical Columns: {categorical_cols}")
    
#     if categorical_cols:
#         encoder = ce.BinaryEncoder(cols=categorical_cols, drop_invariant=True)

#         original_shape = original_data.shape
#         print(f"Original Data Shape: {original_shape}")

#         # Perform encoding and convert to DataFrame with correct column names
#         encoded_array = encoder.fit_transform(original_data)
#         encoded_data = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

#         # Compute the mapping: which new columns came from which original column
#         all_encoded_cols = list(encoded_data.columns)
#         for col in categorical_cols:
#             mapped = [c for c in all_encoded_cols if c.startswith(col + "_")]
#             if mapped:
#                 encoding_map[col] = mapped

#         new_columns = list(set(encoded_data.columns) - set(original_data.columns))
#         print(f"✅ Binary Encoding applied. New Features Added: {new_columns}")
#         print(f"New Data Shape after Encoding: {encoded_data.shape}")

#     else:
#         print("⚠ No categorical columns found. Returning original data.")
#         encoded_data = original_data.copy()

#     return encoded_data, encoding_map
