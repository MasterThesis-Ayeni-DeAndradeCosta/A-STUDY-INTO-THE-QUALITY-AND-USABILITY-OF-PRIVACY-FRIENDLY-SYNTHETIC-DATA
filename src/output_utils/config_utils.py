def generate_anonym_tag(config):
    """
    Generates a unique tag for anonymized files based on configuration.

    Format: k{k}_l{l}_sup{XX}
    Example: k3_l2_sup05
    """
    anon_config = config.get("anonymization", {})
    models = anon_config.get("models", {})

    # Extract individual values
    k = models.get("k_anonymity", "kX")
    l_val = models.get("l_diversity", {}).get("value", "lX")
    sup = anon_config.get("suppression_limit", "supX")

    # Format suppression
    sup_str = f"sup{int(sup * 100):02d}" if isinstance(sup, float) else str(sup)

    # Format l-diversity
    l_tag = f"l{l_val}" if isinstance(l_val, int) else str(l_val)

    return f"k{k}_{l_tag}_{sup_str}"
