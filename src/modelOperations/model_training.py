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

def train_models(datasets):
    """
    Trains predefined models on the given datasets.

    Parameters:
    - datasets (dict): Dictionary with dataset names as keys and (X_train, y_train) tuples as values.

    Returns:
    - trained_models (dict): Dictionary with dataset names as keys and dictionaries of trained models as values.
    """
    print("\nTraining models...")

    # Temporarily suppress warnings generated within this code block
    # TEMPORARY SOLUTION: Suppress certain sklearn warnings to reduce output clutter.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    # Predefined models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=3000),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC()
    }

    trained_models = {dataset_name: {} for dataset_name in datasets.keys()}

    # Train each model on each dataset
    for dataset_name, (X_train, y_train) in datasets.items():
        print(f'\nTraining models on {dataset_name} dataset:')
        for model_name, model in models.items():
            print(f'Training {model_name}...')
            model.fit(X_train, y_train)  # Train the model
            trained_models[dataset_name][model_name] = model  # Store the trained model
            print(f'{model_name} trained successfully.')

    return trained_models

def evaluate_models(trained_models, X_test_original, y_test_original, datasets):
    """
    Evaluates trained models on test data and cross-validation.

    Parameters:
    - trained_models (dict): Dictionary with dataset names as keys and dictionaries of trained models as values.
    - X_test_original (DataFrame): Test features for the original dataset.
    - y_test_original (Series): Test labels for the original dataset.
    - datasets (dict): Dictionary with dataset names as keys and (X_train, y_train) tuples as values.

    Returns:
    - results_df (DataFrame): DataFrame containing evaluation metrics for each model and dataset.
    """
    model_results = []

    # Iterate through each dataset
    for dataset_name, models in trained_models.items():
        print(f'\nEvaluating models trained on {dataset_name} dataset:')

        # Iterate through each model
        for model_name, model in models.items():
            print(f'\nProcessing {model_name}...')

            # Get training data for cross-validation
            X_train, y_train = datasets[dataset_name]

            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            # Predict on the original test set
            predictions = model.predict(X_test_original)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_original,
                predictions,
                average='weighted'
            )

            # Store results
            model_results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': round(np.mean(scores), 4),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1': round(f1, 4)
            })

            print(f'Completed {model_name} evaluation.')

    # Convert results to a DataFrame
    results_df = pd.DataFrame(model_results)

    # Print results for quick inspection
    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))

    return results_df