import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_original_data(data, target_column, test_size=0.2, random_state=42):
    """
    Splits the original dataset into training and test sets.

    Parameters:
    - data (DataFrame): Original dataset.
    - target_column (str): Target variable.
    - test_size (float): Proportion of test data. Default is 0.2.
    - random_state (int): Seed for reproducibility. Default is 42.

    Returns:
    - X_train (DataFrame): Training features.
    - X_test (DataFrame): Test features.
    - y_train (Series): Training labels.
    - y_test (Series): Test labels.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prepare_synthetic_data(data, target_column):
    """
    Prepares synthetic dataset without splitting.

    Parameters:
    - data (DataFrame): Synthetic dataset.
    - target_column (str): Target variable.

    Returns:
    - X (DataFrame): Features.
    - y (Series): Labels.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y


def encode_categorical_features(data, target_column):
    """
    Encodes categorical features in the dataset using one-hot encoding.

    Parameters:
    - data (DataFrame): The dataset to encode.
    - target_column (str): The target column that should not be encoded.

    Returns:
    - data_encoded (DataFrame): The dataset with categorical features encoded.
    """
    # Identify categorical columns, excluding the target column
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != target_column]
    
    # Print the identified categorical columns
    print(f"Identified categorical columns for encoding: {categorical_cols}")

    # If there are no categorical columns, just return the data
    if not categorical_cols:
        print("No categorical columns found. Skipping encoding.")
        return data
    
    # Apply one-hot encoding to categorical columns
    try:
        print("Starting one-hot encoding...")
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        print(f"Encoding complete. Dataset now has {data_encoded.shape[1]} columns.")
    except Exception as e:
        print(f"Error during encoding: {e}")
        raise
    
    return data_encoded
