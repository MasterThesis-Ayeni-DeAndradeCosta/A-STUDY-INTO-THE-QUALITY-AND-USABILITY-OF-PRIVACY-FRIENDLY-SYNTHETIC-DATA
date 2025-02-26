from abc import ABC, abstractmethod
import pandas as pd
import pickle

class BaseSynthesizer(ABC):
    """
    Abstract base class for all synthesizers.
    Every synthesizer must implement `fit()` and `sample()`.
    """

    def __init__(self, metadata, **kwargs):
        self.metadata = metadata
        self.params = kwargs

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Trains the synthesizer on the provided dataset."""
        pass

    @abstractmethod
    def sample(self, num_rows: int) -> pd.DataFrame:
        """Generates synthetic data."""
        pass

    def save(self, path):
        """Saves the synthesizer instance using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Synthesizer saved at {path}")

    @staticmethod
    def load(path):
        """Loads a synthesizer instance from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)
