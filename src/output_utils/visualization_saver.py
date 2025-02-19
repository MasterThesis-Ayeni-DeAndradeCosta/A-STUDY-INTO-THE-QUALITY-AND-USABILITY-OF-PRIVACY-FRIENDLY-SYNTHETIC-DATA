import os
import matplotlib.pyplot as plt

def save_model_performance_graph(output_dir, results_df, dataset_name):
    """
    Saves model performance metrics as a bar chart.

    Parameters:
    - output_dir (str): Directory where the visualization will be saved.
    - results_df (DataFrame): DataFrame containing model evaluation metrics.
    - dataset_name (str): Name of the dataset.
    """
    model_names = results_df["Model"].unique()
    accuracy_scores = results_df.groupby("Model")["Accuracy"].mean().values

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, accuracy_scores, color="skyblue")
    plt.ylabel("Accuracy")
    plt.title(f"Model Accuracy Comparison ({dataset_name})")
    plt.xticks(rotation=45)

    image_path = os.path.join(output_dir, f"{dataset_name}_model_accuracy.png")
    plt.savefig(image_path)
    plt.close()

    print(f"📊 Model performance graph saved at {image_path}")
