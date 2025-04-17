import os
import pandas as pd
from preprocessing.encoding import encode_categorical_features

def run_postprocessing(anonymized_path, separator, target_column, encoder=None, logger=None):
    """
    Post-processes the anonymized dataset by encoding it only.

    Returns:
    - anonymized_encoded_df (DataFrame): Encoded anonymized DataFrame.
    - encoded_output_path (str): Path to saved encoded anonymized CSV.
    """
    if anonymized_path is None or not os.path.exists(anonymized_path):
        print("⚠️ No anonymized data found. Skipping postprocessing.")
        if logger:
            logger.warning("No anonymized data found. Skipping postprocessing.")
        return None, None, None

    dataset_name = os.path.splitext(os.path.basename(anonymized_path))[0].replace("_anonymized", "")
    base_dir = os.path.dirname(anonymized_path)
    encoded_output_path = os.path.join(base_dir, f"{dataset_name}_anonymized_encoded.csv")

    #encoded_output_path = f"datasets/anonymized/{dataset_name}_anonymized_encoded.csv"

    # Load and encode anonymized data
    df_anonymized = pd.read_csv(anonymized_path, sep=separator)
    print(f"\n📂 Loaded anonymized data: {df_anonymized.shape}")
    if logger:
        logger.info(f"Loaded anonymized data: {df_anonymized.shape}")
    df_encoded, encoding_map = encode_categorical_features(df_anonymized, target_column, encoder=encoder)
    print(f"✅ Encoding applied to anonymized data. Encoded shape: {df_encoded.shape}")
    if logger:
        logger.info(f"Encoding applied to anonymized data. Encoded shape: {df_encoded.shape}")

    # Save encoded anonymized data
    os.makedirs(os.path.dirname(encoded_output_path), exist_ok=True)
    df_encoded.to_csv(encoded_output_path, index=False)
    print(f"✅ Encoded anonymized dataset saved to: {encoded_output_path}")
    if logger:
        logger.info(f"Encoded anonymized dataset saved to: {encoded_output_path}")

    return df_encoded, encoded_output_path, encoding_map


if __name__ == "__main__":
    print("This script is not meant to be run directly.")
