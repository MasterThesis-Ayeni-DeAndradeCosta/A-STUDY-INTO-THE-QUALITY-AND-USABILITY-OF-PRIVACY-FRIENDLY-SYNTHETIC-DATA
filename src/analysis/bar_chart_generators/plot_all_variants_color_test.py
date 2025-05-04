
import matplotlib.pyplot as plt
import os

def plot_all_variants(model_name, dataset_name, metrics_dict, show_settings, save_dir, suffix=""):
    """
    Plots a grouped bar chart comparing Original, Anonymized, Synthetic, and Hybrid metrics.

    Parameters:
    - model_name (str)
    - dataset_name (str)
    - metrics_dict (dict): mapping label -> {metric_name -> value}
    - show_settings (dict): which metrics to include
    - save_dir (str): where to save
    - suffix (str): optional file suffix
    """
   
    colors = {
        "Original": "#27AE60",         
        "Anonymous": "#9B59B6",        
        "CTGAN": "#2980B9",            
        "TVAE": "#3498DB",             
        "GaussianCopula": "#5DADE2",   
        "CTGAN_HYBRID": "#6C5CE7",    
        "TVAE_HYBRID":"#7D5FFF" ,      
        "GaussianCopula_HYBRID": "#A29BFE"  
    }

    metrics = [m for m in show_settings if show_settings[m]]
    variants = list(metrics_dict.keys())
    x = range(len(metrics))
    bar_width = 0.8 / len(variants)  # dynamic width

    plt.figure(figsize=(14, 6))
    
    for i, variant in enumerate(variants):
        values = [metrics_dict[variant][m] for m in metrics]
        x_shifted = [pos + i * bar_width - (bar_width * (len(variants)-1) / 2) for pos in x]
        plt.bar(x_shifted, values, width=bar_width, label=variant, color=colors[variant], edgecolor="black")

        # Add value labels
        for xi, val in zip(x_shifted, values):
            plt.text(xi, val + 0.01, f"{val:.2f}", ha='center', fontsize=8)

    plt.xticks(ticks=x, labels=metrics, rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 1)
    plt.ylabel("Score", fontsize=12)
    plt.title(f"{model_name} on {dataset_name} — All Variants", fontsize=14, weight="bold")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save chart
    filename = f"{model_name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_all_variants"
    if suffix:
        filename += f"_{suffix}"
    filename += ".png"

    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300)
    print(f"✅ Chart saved to: {full_path}")

    plt.show()


if __name__ == "__main__":
    model = "Random Forest"
    dataset = "Loan Dataset"

    dummy = lambda base: { "Accuracy": base, "Precision": base + 0.02, "Recall": base, "F1": base - 0.01, "AUC-ROC": base + 0.015 }

    metrics_by_variant = {
        "Original": dummy(0.84),
        "Anonymous": dummy(0.78),
        "CTGAN": dummy(0.75),
        "TVAE": dummy(0.73),
        "GaussianCopula": dummy(0.72),
        "CTGAN_HYBRID": dummy(0.76),
        "TVAE_HYBRID": dummy(0.74),
        "GaussianCopula_HYBRID": dummy(0.735),
    }

    show_metrics = {
        "Accuracy": True,
        "Precision": True,
        "Recall": True,
        "F1": True,
        "AUC-ROC": True
    }

    save_dir = r"C:\Users\delea\OneDrive\Documents\Desktop\Master_Thesis\Semester 2\writing the thesis\graphs\loan\summary"
    suffix = "color_test"

    plot_all_variants(model, dataset, metrics_by_variant, show_metrics, save_dir, suffix)

