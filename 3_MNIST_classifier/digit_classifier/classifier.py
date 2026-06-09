import numpy as np

from digit_classifier.interface import DigitClassificationInterface
from digit_classifier.models import CNNModel, RandomForestModel, RandomModel


class DigitClassifier:
    """
    Facade class for digit classification.

    Supported algorithms:
    - cnn
    - rf
    - rand
    """

    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.model = self._create_model(algorithm)

    def _create_model(self, algorithm: str) -> DigitClassificationInterface:
        models = {
            "cnn": CNNModel,
            "rf": RandomForestModel,
            "rand": RandomModel,
        } # extend while adding new models

        if algorithm not in models:
            available = ", ".join(models.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Available algorithms: {available}."
            )

        return models[algorithm]()

    def predict(self, image: np.ndarray) -> int:
        self._validate_input(image)

        prediction = self.model.predict(image)

        if not isinstance(prediction, int):
            raise TypeError("Prediction must be an integer.")

        if not 0 <= prediction <= 9:
            raise ValueError("Prediction must be between 0 and 9.")

        return prediction

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        if image.shape != (28, 28, 1):
            raise ValueError("Input image must have shape (28, 28, 1).")