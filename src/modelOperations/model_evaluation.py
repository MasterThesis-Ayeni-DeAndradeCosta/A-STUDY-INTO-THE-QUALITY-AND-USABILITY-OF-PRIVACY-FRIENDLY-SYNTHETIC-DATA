import sys
import os
import importlib
import pandas as pd
from modelOperations.evaluation_metrics import *


# Get the absolute path of the `src` directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_metrics(config):
    """Dynamically loads evaluation metrics based on YAML config."""
    metrics = {}
    metric_classes = {
        "Accuracy": AccuracyMetric,
        "Precision": PrecisionMetric,
        "Recall": RecallMetric,
        "F1": F1Metric,
        "AUC-ROC": AUCRocMetric,
        "LogLoss": LogLossMetric,
        "CohenKappa": CohenKappaMetric,
        "MCC": MCCMetric,
    }

    for metric_name, metric_class in metric_classes.items():
        if config["utility"]["metrics"].get(metric_name, False):  # Load only enabled metrics
            metrics[metric_name] = metric_class()

    return metrics

def evaluate_models(trained_models, X_test_original, y_test_original, datasets, config):
    """
    Evaluates trained models using dynamically loaded metrics from configuration.

    Parameters:
    - trained_models (dict): Dictionary of trained models.
    - X_test_original (DataFrame): Original test data.
    - y_test_original (Series): Original test labels.
    - datasets (dict): Dictionary containing datasets.
    - config (dict): Configuration dictionary.

    Returns:
    - results_df (DataFrame): Evaluation metrics DataFrame for each model-dataset combination.
    """
    model_results = []
    enabled_metrics = load_metrics(config)  # Load only enabled metrics dynamically

    model_names = next(iter(trained_models.values())).keys() if trained_models else []

    for model_name in model_names:
        for dataset_name in datasets.keys():
            if dataset_name in trained_models and model_name in trained_models[dataset_name]:
                model = trained_models[dataset_name][model_name]
                print(f'\nEvaluating {model_name} on {dataset_name}...')
                y_pred = model.predict(X_test_original)

                metric_values = {}

                for metric_name, metric_obj in enabled_metrics.items():
                    metric_values[metric_name] = metric_obj.compute(y_test_original, y_pred, model, X_test_original)

                # Store results
                model_results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    **metric_values
                })

    results_df = pd.DataFrame(model_results).sort_values(by=["Model", "Dataset"])

    # Print results
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))

    return results_df
