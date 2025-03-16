import os
import pandas as pd
import yaml
from preprocessing.encoding import encode_categorical_features

def load_config(config_path="configs/benchmark_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_postprocessing(anonymized_path=None):
    config = load_config()
    dataset_path = config["dataset"]["path"]
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Use default path if none provided
    if anonymized_path is None:
        anonymized_path = f"datasets/anonymized/{dataset_name}_anonymized.csv"
    
    encoded_output_path = f"datasets/anonymized/{dataset_name}_anonymized_encoded.csv"
    target_column = config["dataset"]["target_column"]
    encoding_type = config["preprocessing"]["encoding_type"]

    # Load anonymized data
    df = pd.read_csv(anonymized_path)
    print(f"\n📂 Loaded anonymized data: {df.shape}")

    # Apply encoding
    df_encoded = encode_categorical_features(df, target_column)
    print(f"✅ Encoding applied. Encoded shape: {df_encoded.shape}")

    # Save result
    os.makedirs(os.path.dirname(encoded_output_path), exist_ok=True)
    df_encoded.to_csv(encoded_output_path, index=False)
    print(f"✅ Encoded anonymized dataset saved to: {encoded_output_path}")

if __name__ == "__main__":
    run_postprocessing()
