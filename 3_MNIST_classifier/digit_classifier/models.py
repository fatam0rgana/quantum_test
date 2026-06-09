import random
import numpy as np

from digit_classifier.interface import DigitClassificationInterface


class CNNModel(DigitClassificationInterface):
    """
    CNN model adapter.

    Expected internal input:
    - tensor with shape: 28x28x1

    For this task, real CNN inference is mocked.
    """

    def __init__(self):
        self.model = None
        # self.model = keras.models.load_model("cnn_model.keras")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError("CNNModel expects image with shape (28, 28, 1).")

        # Placeholder for real CNN inference.
        # Example:
        # prediction = self.model.predict(image[np.newaxis, ...])
        # return int(np.argmax(prediction))
        if self.model is None:
            return 0

        pred = self.model.predict(image[np.newaxis, ...])
        return int(np.argmax(pred))


class RandomForestModel(DigitClassificationInterface):
    """
    Random Forest model adapter.

    Expected internal input:
    - flattened array with shape: (784,)

    For this task, real RF inference is mocked.
    """
    def __init__(self):
        self.model = None
        # self.model = joblib.load("rf_model.joblib")

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError("RandomForestModel expects image with shape (28, 28, 1).")

        flattened_image = image.reshape(-1)

        if flattened_image.shape != (784,):
            raise ValueError("RandomForestModel expects flattened image of length 784.")

        # Placeholder for real Random Forest inference.
        # Example:
        # prediction = self.model.predict([flattened_image])
        # return int(prediction[0])
        if self.model is None:
            return 0
        return int(self.model.predict([flattened_image])[0])


class RandomModel(DigitClassificationInterface):
    """
    Random model.

    Expected internal input:
    - center crop with shape: 10x10

    Returns random digit from 0 to 9.
    """

    def predict(self, image: np.ndarray) -> int:
        if image.shape != (28, 28, 1):
            raise ValueError("RandomModel expects image with shape (28, 28, 1).")

        center_crop = image[9:19, 9:19, 0]

        if center_crop.shape != (10, 10):
            raise ValueError("RandomModel expects center crop with shape (10, 10).")

        return random.randint(0, 9)