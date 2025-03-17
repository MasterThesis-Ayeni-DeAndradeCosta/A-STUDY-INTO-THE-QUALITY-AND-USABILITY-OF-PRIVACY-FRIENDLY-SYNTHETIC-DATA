import os
import pandas as pd
from preprocessing.encoding import encode_categorical_features


def run_postprocessing(anonymized_path, separator, target_column):
    dataset_name = os.path.splitext(os.path.basename(anonymized_path))[0].replace("_anonymized", "")
    
    encoded_output_path = f"datasets/anonymized/{dataset_name}_anonymized_encoded.csv"

    # Load anonymized data
    df = pd.read_csv(anonymized_path, sep=separator)
    print(f"\n📂 Loaded anonymized data: {df.shape}")

    # Apply encoding
    df_encoded = encode_categorical_features(df, target_column)
    print(f"✅ Encoding applied. Encoded shape: {df_encoded.shape}")

    # Save result
    os.makedirs(os.path.dirname(encoded_output_path), exist_ok=True)
    df_encoded.to_csv(encoded_output_path, index=False)
    print(f"✅ Encoded anonymized dataset saved to: {encoded_output_path}")

    return df_encoded

