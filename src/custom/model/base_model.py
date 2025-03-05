from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import pandas as pd

class BaseMLModel(ABC):
    """Abstract base class for ML models."""

    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions on the test set."""
        pass
