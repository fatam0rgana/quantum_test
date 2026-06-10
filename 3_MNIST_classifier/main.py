import numpy as np
from loguru import logger

from digit_classifier import DigitClassifier


def main():
    algorithm = input(
        "Select algorithm (cnn, rf, rand): "
    ).strip().lower()

    image = np.random.rand(28, 28, 1)

    classifier = DigitClassifier(algorithm)
    prediction = classifier.predict(image)

    logger.info(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()