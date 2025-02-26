import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_model_performance(results_df, dataset_name, output_dir="outputs"):
    """
    Visualizes model performance metrics for original and multiple synthetic datasets.

    Parameters:
    - results_df (DataFrame): DataFrame containing model evaluation metrics.
    - dataset_name (str): Name of the dataset being visualized.
    - output_dir (str): Directory where the figure will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Get all unique model names and dataset names
    model_names = results_df['Model'].unique()
    dataset_names = results_df['Dataset'].unique()  

    # Initialize dictionaries for accuracy and precision
    accuracy_scores = {dataset: [] for dataset in dataset_names}
    precision_scores = {dataset: [] for dataset in dataset_names}

    # Collect performance metrics for each model
    for model in model_names:
        for dataset in dataset_names:
            dataset_results = results_df[(results_df['Dataset'] == dataset) & (results_df['Model'] == model)]
            accuracy = dataset_results['Accuracy'].values.tolist()
            precision = dataset_results['Precision'].values.tolist()

            accuracy_scores[dataset].append(accuracy[0] if accuracy else None)
            precision_scores[dataset].append(precision[0] if precision else None)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))

    x = np.arange(len(model_names))
    width = max(0.2 / len(dataset_names), 0.05)  # Ensures bars remain visible

    # Dynamically generate colors
    cmap = plt.get_cmap("tab10")  
    colors = [cmap(i) for i in range(len(dataset_names))]

    # Plot accuracy
    for i, dataset in enumerate(dataset_names):
        ax1.bar(x + (i - len(dataset_names)/2) * width, accuracy_scores[dataset], width, label=dataset, color=colors[i])
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison Across Datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()

    # Plot precision
    for i, dataset in enumerate(dataset_names):
        ax2.bar(x + (i - len(dataset_names)/2) * width, precision_scores[dataset], width, label=dataset, color=colors[i])
    
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision Comparison Across Datasets')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()

    plt.suptitle(f'Model Performance Across Original and Synthetic Data for {dataset_name}', fontsize=14)
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(output_dir, f"{dataset_name}_model_performance.png")
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved at {save_path}")

    plt.show()

    # Print detailed performance metrics
    print("\nDetailed Model Performance Comparison:")
    print(results_df.to_string(index=False))
