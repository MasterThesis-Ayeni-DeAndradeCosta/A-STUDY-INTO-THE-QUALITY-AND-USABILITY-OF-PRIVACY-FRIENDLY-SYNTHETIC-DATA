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

def evaluate_models(trained_models, X_test_original, y_test_original, datasets, config, logger=None):
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
        if logger:
            logger.info(f"Preparing evaluation for model '{model_name}' across all datasets...")
        print(f"\nPreparing evaluation for model '{model_name}' across all datasets...")
        for dataset_name, model_dict in trained_models.items():
            if model_name in model_dict:
                model = model_dict[model_name]
                print(f'\nEvaluating {model_name} on {dataset_name}...')
                if logger:
                    logger.info(f"Evaluating {model_name} on {dataset_name}...")

                # Align X_test_original to model training features
                training_features = model.model.feature_names_in_  # Features seen during fit
                X_test_aligned = X_test_original.loc[:, X_test_original.columns.isin(training_features)]

                # Reindex to ensure correct column order
                X_test_aligned = X_test_aligned.reindex(columns=training_features, fill_value=0)

                if logger and X_test_aligned.isnull().values.any():
                    logger.warning(f" [WARNING] NaNs found in X_test_aligned for model {model_name} on {dataset_name}")
                    

                # Predict using aligned test set
                #y_pred = model.predict(X_test_aligned)
                try:
                    y_pred = model.predict(X_test_aligned)
                    if logger:
                        logger.info(f"Predictions completed for {model_name} on {dataset_name}")
                        logger.info(f"Prediction distribution for {model_name} on {dataset_name}: {pd.Series(y_pred).value_counts().to_dict()}") 
                    if len(set(y_pred)) < 2 and logger:
                        logger.warning(f" [WARNING] :{model_name} on {dataset_name} predicted only one class: {set(y_pred)}")
                except Exception as e:
                    if logger:
                        logger.error(f" [ERROR]: Prediction failed for {model_name} on {dataset_name}: {e}")
                    continue

                metric_values = {}
                for metric_name, metric_obj in enabled_metrics.items():
                    #metric_values[metric_name] = metric_obj.compute(y_test_original, y_pred, model, X_test_aligned)
                    #more safe version
                    try:
                        value = metric_obj.compute(y_test_original, y_pred, model, X_test_aligned)
                        metric_values[metric_name] = value
                    except Exception as e:
                        metric_values[metric_name] = "N/A"
                        if logger:
                            logger.warning(f" Failed to compute {metric_name} for {model_name} on {dataset_name}: {e}")

                if logger:
                    logger.info(f"Computed {len(metric_values)} metric(s) for {model_name} on {dataset_name}") 
                    
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
    if logger:
        logger.info("Evaluation completed for all models.")

    return results_df
