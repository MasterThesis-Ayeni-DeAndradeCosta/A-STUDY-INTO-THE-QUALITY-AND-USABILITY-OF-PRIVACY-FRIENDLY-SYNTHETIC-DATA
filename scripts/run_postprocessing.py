import os
import pandas as pd
from preprocessing.encoding import encode_categorical_features
from anon_postprocessing_pipeline.normalise import normalise_dataframe, build_mode_maps

def run_postprocessing(anonymized_path, separator, target_column, train_raw_path, encoder=None, logger=None):
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
        return None, None, None, None
    
    if logger:
        logger.info(f" [POSTPROCESSING] Postprocessing anonymized data: {anonymized_path}")

    dataset_name = os.path.splitext(os.path.basename(anonymized_path))[0].replace("_anonymized", "")
    base_dir = os.path.dirname(anonymized_path)
    encoded_output_path = os.path.join(base_dir, f"{dataset_name}_anonymized_encoded.csv")

    
    
    # Load and encode anonymized data
    df_anonymized = pd.read_csv(anonymized_path, sep=separator)
    print(f"\n📂 Loaded anonymized data: {df_anonymized.shape}")
    if logger:
        logger.info(f" [POSTPROCESSING] Loaded anonymized data: {df_anonymized.shape}")

    #additional steps to make sure the same encoder can be used
    #Load train_raw explicitly to calculate fallback values
    train_raw_df = pd.read_csv(train_raw_path, sep=separator)
    cat_modes, cat_uniques, num_modes = build_mode_maps(train_raw_df)
    if logger:
        logger.info(f" [POSTPROCESSING] Loaded fallback maps from training data. Categorical cols: {len(cat_modes)}, Numeric cols: {len(num_modes)}")

    if logger:
        logger.info(" [POSTPROCESSING] Normalizing anonymized data using ARX artifact cleaning and fallback values.")
    # Normalize anonymized data
    df_cleaned = normalise_dataframe(df_anonymized, cat_modes, cat_uniques, num_modes)

    processed_output_path = os.path.join(base_dir, f"{dataset_name}_anonymized_processed.csv")
    df_cleaned.to_csv(processed_output_path, index=False, sep=separator)
    if logger:
        logger.info(f" [POSTPROCESSING] Normalized anonymized data saved to: {processed_output_path}")

    # Encode using provided encoder
    if encoder is None:
        print(" No encoder present. Skipping encoding of anonymized data.")
        if logger:
            logger.info(" [POSTPROCESSING] No encoder present. Skipping encoding of anonymized data.Will use processed anonymous data directly")
        return df_cleaned, processed_output_path, None, {}

    
    df_encoded, encoding_map = encode_categorical_features(df_cleaned, target_column, encoder=encoder)
    

    #df_encoded, encoding_map = encode_categorical_features(df_anonymized, target_column, encoder=encoder)
    print(f"✅ Binary Encoding applied to anonymized data. Encoded shape: {df_encoded.shape}")
    if logger:
        logger.info(f" [POSTPROCESSING] Binary Encoding applied to anonymized data. Encoded shape: {df_encoded.shape}")

    # Save encoded anonymized data
    os.makedirs(os.path.dirname(encoded_output_path), exist_ok=True)
    df_encoded.to_csv(encoded_output_path, index=False, sep=separator)
    print(f" Encoded anonymized dataset saved to: {encoded_output_path}")
    if logger:
        logger.info(f" [POSTPROCESSING] Encoded anonymized dataset saved to: {encoded_output_path}")
        logger.info(f" [POSTPROCESSING] Encoding map generated for anonymized data: {len(encoding_map)} original categorical columns.")

    return df_encoded, processed_output_path, encoded_output_path, encoding_map


if __name__ == "__main__":
    print("This script is not meant to be run directly.")
