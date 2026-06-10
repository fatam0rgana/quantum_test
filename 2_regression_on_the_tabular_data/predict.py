import os
import joblib
import pandas as pd
from loguru import logger

from data_preparation import DataPreprocessor

class RegressionPredictor:
    """Load trained model and generate predictions."""

    def __init__(
        self,
        test_path="datasets/hidden_test.csv",
        model_path="models/hgb_model.joblib",
        output_path="predictions/predictions.csv",
    ):
        self.test_path = test_path
        self.model_path = model_path
        self.output_path = output_path

    def load_data(self):
        return pd.read_csv(self.test_path)

    def load_model(self):
        return joblib.load(self.model_path)

    def save_predictions(self, test_data, predictions):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        output = test_data.copy()
        output["prediction"] = predictions

        output.to_csv(self.output_path, index=False)

        logger.info(f"Predictions saved to: {self.output_path}")

    def run(self):
        data = self.load_data()
        model = self.load_model()
        #data = DataPreprocessor().transform(data)
        predictions = model.predict(data)

        self.save_predictions(data, predictions)


if __name__ == "__main__":
    predictor = RegressionPredictor()
    predictor.run()