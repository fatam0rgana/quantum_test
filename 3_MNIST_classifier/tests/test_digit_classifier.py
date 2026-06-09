import unittest
import numpy as np

from digit_classifier import DigitClassifier
from digit_classifier.models import CNNModel, RandomForestModel, RandomModel
from digit_classifier.interface import DigitClassificationInterface


class TestDigitClassifier(unittest.TestCase):

    def setUp(self):
        self.image = np.random.rand(28, 28, 1)

    def test_cnn_classifier_returns_integer(self):
        classifier = DigitClassifier("cnn")
        prediction = classifier.predict(self.image)

        self.assertIsInstance(prediction, int)
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 9)

    def test_rf_classifier_returns_integer(self):
        classifier = DigitClassifier("rf")
        prediction = classifier.predict(self.image)

        self.assertIsInstance(prediction, int)
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 9)

    def test_random_classifier_returns_integer(self):
        classifier = DigitClassifier("rand")
        prediction = classifier.predict(self.image)

        self.assertIsInstance(prediction, int)
        self.assertGreaterEqual(prediction, 0)
        self.assertLessEqual(prediction, 9)

    def test_unknown_algorithm_raises_error(self):
        with self.assertRaises(ValueError):
            DigitClassifier("xgboost")

    def test_invalid_input_shape_raises_error(self):
        classifier = DigitClassifier("cnn")
        invalid_image = np.random.rand(28, 28)

        with self.assertRaises(ValueError):
            classifier.predict(invalid_image)

    def test_invalid_input_type_raises_error(self):
        classifier = DigitClassifier("cnn")

        with self.assertRaises(TypeError):
            classifier.predict([[0] * 28] * 28)

    def test_models_implement_interface(self):
        self.assertIsInstance(CNNModel(), DigitClassificationInterface)
        self.assertIsInstance(RandomForestModel(), DigitClassificationInterface)
        self.assertIsInstance(RandomModel(), DigitClassificationInterface)


if __name__ == "__main__":
    unittest.main()