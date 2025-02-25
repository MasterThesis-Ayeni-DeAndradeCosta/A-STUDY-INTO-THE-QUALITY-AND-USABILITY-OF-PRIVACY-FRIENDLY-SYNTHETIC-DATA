# Required imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Required imports
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


def train_models(datasets, config):
    """
    Trains predefined models on the given datasets.

    Parameters:
    - datasets (dict): Dictionary with dataset names as keys and (X_train, y_train) tuples as values.
    - config (dict): Configuration dictionary containing model selection.

    Returns:
    - trained_models (dict): Dictionary with dataset names as keys and dictionaries of trained models as values.
    """
    print("\nTraining models...")

    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    # Get model selection from YAML
    selected_models = config["utility"]["models"]

    # Dynamically initialize models
    models = {}
    if selected_models.get("LogisticRegression", False):
        models["Logistic Regression"] = LogisticRegression(max_iter=3000)
    if selected_models.get("KNN", False):
        models["KNN"] = KNeighborsClassifier()
    if selected_models.get("RandomForest", False):
        models["Random Forest"] = RandomForestClassifier()
    if selected_models.get("DecisionTree", False):
        models["Decision Tree"] = DecisionTreeClassifier()
    if selected_models.get("SVM", False):
        models["SVM"] = SVC()

    trained_models = {}

    # Train models on each dataset (including synthetic ones)
    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        print(f'\nTraining models on {dataset_name} dataset:')
        trained_models[dataset_name] = {}
        for model_name, model in models.items():
            print(f'Training {model_name} on {dataset_name}...')
            model.fit(X_train, y_train)  # Train the model
            trained_models[dataset_name][model_name] = model  # Store trained model
            print(f'{model_name} trained successfully on {dataset_name}.')

    return trained_models


def evaluate_models(trained_models, X_test_original, y_test_original, datasets):
    """
    Evaluates trained models on corresponding test data.

    Parameters:
    - trained_models (dict): Dictionary with dataset names as keys and dictionaries of trained models as values.
    - datasets (dict): Dictionary with dataset names as keys and (X_train, X_test, y_train, y_test) tuples as values.

    Returns:
    - results_df (DataFrame): DataFrame containing evaluation metrics for each model and dataset.
    """
    model_results = []

    # Iterate through datasets
    for dataset_name, models in trained_models.items():
        print(f'\nEvaluating models trained on {dataset_name} dataset...')

        # Get test set corresponding to this dataset
        X_test, y_test = X_test_original, y_test_original

        # Evaluate each model
        for model_name, model in models.items():
            print(f'\nProcessing {model_name} on {dataset_name}...')

            # Cross-validation score
            scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

            # Predict on the test set
            predictions = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predictions, average='weighted'
            )

            # Store results
            model_results.append({
                'Dataset': dataset_name,  # should say "CTGAN Data" or "TVAE Data"
                'Model': model_name,
                'Accuracy': round(np.mean(scores), 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1': round(f1, 4)
            })

            print(f'Completed {model_name} evaluation on {dataset_name}.')

    # Convert results to a DataFrame
    results_df = pd.DataFrame(model_results)

    # Print results for quick inspection
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))

    return results_df
