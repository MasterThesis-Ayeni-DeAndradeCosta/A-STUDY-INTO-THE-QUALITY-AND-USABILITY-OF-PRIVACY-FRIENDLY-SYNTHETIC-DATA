from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    Every model must implement `train()` and `predict()`.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the model on the provided dataset."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Generates predictions using the trained model."""
        pass
