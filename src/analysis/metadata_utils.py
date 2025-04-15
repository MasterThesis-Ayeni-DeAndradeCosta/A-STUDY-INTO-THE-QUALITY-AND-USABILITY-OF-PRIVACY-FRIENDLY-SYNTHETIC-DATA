import yaml

def extract_config_metadata(config_path, dataset_type):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    meta = {}
    meta["test_size"] = config.get("dataset", {}).get("test_size")

    if dataset_type == "Anonymous":
        anon = config.get("anonymization", {})
        meta["k_anonymity"] = anon.get("models", {}).get("k_anonymity")
        meta["l_diversity"] = anon.get("models", {}).get("l_diversity", {}).get("value")
        meta["suppression_limit"] = anon.get("suppression_limit")
        meta["epochs"] = None
        meta["custom_generated_rows"] = None

    elif dataset_type in ["CTGAN", "TVAE", "GaussianCopula", "Hybrid"]:
        meta["k_anonymity"] = meta["l_diversity"] = meta["suppression_limit"] = None
        for synth_name, synth_conf in config.get("synthesis", {}).get("synthesizers", {}).items():
            if synth_conf.get("enabled"):
                meta["epochs"] = synth_conf.get("params", {}).get("epochs")
                meta["custom_generated_rows"] = synth_conf.get("custom_generated_rows")
                break
        else:
            meta["epochs"] = meta["custom_generated_rows"] = None
    else:
        meta["k_anonymity"] = meta["l_diversity"] = meta["suppression_limit"] = None
        meta["epochs"] = meta["custom_generated_rows"] = None

    return meta
