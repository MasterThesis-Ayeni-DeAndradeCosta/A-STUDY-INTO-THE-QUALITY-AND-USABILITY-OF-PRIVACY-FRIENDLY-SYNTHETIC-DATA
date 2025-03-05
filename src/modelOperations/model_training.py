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
from sklearn.metrics import (
    roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef,
    accuracy_score
)
import pickle
import os
from .model_registry import MODEL_REGISTRY

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",  "artifacts", "models"))  # Save models inside artifacts/
os.makedirs(MODEL_DIR, exist_ok=True)

# def train_models(datasets, config):
#     """
#     Trains predefined models on the given datasets.

#     Parameters:
#     - datasets (dict): Dictionary with dataset names as keys and (X_train, y_train) tuples as values.
#     - config (dict): Configuration dictionary containing model selection.

#     Returns:
#     - trained_models (dict): Dictionary with dataset names as keys and dictionaries of trained models as values.
#     """
#     print("\nTraining models...")

#     # Suppress warnings for cleaner output
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=ConvergenceWarning)
#         warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
#     # Get model selection from YAML
#     selected_models = config["utility"]["models"]

#     # Dynamically initialize models
#     models = {}
#     if selected_models.get("LogisticRegression", False):
#         models["Logistic Regression"] = LogisticRegression(max_iter=3000)
#     if selected_models.get("KNN", False):
#         models["KNN"] = KNeighborsClassifier()
#     if selected_models.get("RandomForest", False):
#         models["Random Forest"] = RandomForestClassifier()
#     if selected_models.get("DecisionTree", False):
#         models["Decision Tree"] = DecisionTreeClassifier()
#     if selected_models.get("SVM", False):
#         models["SVM"] = SVC()

#     trained_models = {}

#     # Train models on each dataset (including synthetic ones)
#     for dataset_name, (X_train, y_train) in datasets.items():
#         print(f'\nTraining models on {dataset_name} dataset:')
#         trained_models[dataset_name] = {}
#         for model_name, model in models.items():
#             print(f'Training {model_name} on {dataset_name}...')
#             model.fit(X_train, y_train)  # Train the model
#             trained_models[dataset_name][model_name] = model  # Store trained model
#             print(f'{model_name} trained successfully on {dataset_name}.')

#     return trained_models

def train_models(datasets, config):
    print("\nTraining models from scratch...")
    selected_models = config["utility"]["models"]
    trained_models = {}

    for dataset_name, (X_train, y_train) in datasets.items():
        trained_models[dataset_name] = {}

        for model_name, is_enabled in selected_models.items():
            if is_enabled and model_name in MODEL_REGISTRY:
                print(f"\nTraining {model_name} on {dataset_name}...")

                # Always create a fresh model instance (no loading from file)
                model_instance = MODEL_REGISTRY[model_name]()
                model_instance.train(X_train, y_train)

                trained_models[dataset_name][model_name] = model_instance

    return trained_models
