from abc import ABC, abstractmethod
import numpy as np


class DigitClassificationInterface(ABC):
    """Base interface for all digit classification models."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Predict digit class from input image."""
        raise NotImplementedError

    def train(self, *args, **kwargs):
        """Training is intentionally not implemented for this task."""
        raise NotImplementedError("Training is not implemented.")