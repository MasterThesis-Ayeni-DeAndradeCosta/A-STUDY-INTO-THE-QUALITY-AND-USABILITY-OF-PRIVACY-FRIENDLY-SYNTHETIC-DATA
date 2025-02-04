import numpy as np
import matplotlib.pyplot as plt

def visualize_model_performance(results_df, dataset_name):
    """
    Visualizes model performance metrics for original and synthetic datasets.

    Parameters:
    - results_df (DataFrame): DataFrame containing model evaluation metrics.
    - dataset_name (str): Name of the dataset being visualized.
    """
    model_names = []
    original_accuracy_scores = []
    original_precision_scores = []
    synthetic_accuracy_scores = []
    synthetic_precision_scores = []

    # Separate original and synthetic results
    original_results = results_df[results_df['Dataset'] == 'Original']
    synthetic_results = results_df[results_df['Dataset'] == 'Synthetic']

    for model in original_results['Model'].unique():
        model_names.append(model)
        original_accuracy_scores.append(original_results[original_results['Model'] == model]['Accuracy'].values[0])
        original_precision_scores.append(original_results[original_results['Model'] == model]['Precision'].values[0])
        synthetic_accuracy_scores.append(synthetic_results[synthetic_results['Model'] == model]['Accuracy'].values[0])
        synthetic_precision_scores.append(synthetic_results[synthetic_results['Model'] == model]['Precision'].values[0])

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))

    x = np.arange(len(model_names))
    width = 0.35

    # Accuracy plot
    ax1.bar(x - width/2, original_accuracy_scores, width, label='Original Data', color='skyblue')
    ax1.bar(x + width/2, synthetic_accuracy_scores, width, label='Synthetic Data', color='lightcoral')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()

    # Precision plot
    ax2.bar(x - width/2, original_precision_scores, width, label='Original Data', color='skyblue')
    ax2.bar(x + width/2, synthetic_precision_scores, width, label='Synthetic Data', color='lightcoral')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    ax2.legend()

    plt.suptitle(f'Model Performance: Original vs Synthetic Data for {dataset_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Print detailed performance metrics
    print("\nDetailed Model Performance Comparison:")
    print(results_df.to_string(index=False))
