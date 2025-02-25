from .base_synthesizer import BaseSynthesizer
import pandas as pd

class CustomSynthesizer(BaseSynthesizer):
    """
    A simple example synthesizer that follows the BaseSynthesizer interface.
    """

    def __init__(self, metadata, noise_factor=0.1):
        super().__init__(metadata)
        self.noise_factor = noise_factor

    def fit(self, data: pd.DataFrame):
        print(f"Training CustomSynthesizer with noise_factor={self.noise_factor}...")

    def sample(self, num_rows: int) -> pd.DataFrame:
        print(f"Generating {num_rows} synthetic rows using CustomSynthesizer...")
        return pd.DataFrame({"Synthetic_Column": [1] * num_rows})  # Dummy synthetic data

