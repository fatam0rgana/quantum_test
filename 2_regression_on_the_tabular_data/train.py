import os
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from loguru import logger

from data_preparation import DataPreprocessor

class RegressionTrainer:
    """Train and save regression model."""

    def __init__(
        self,
        train_path="datasets/train.csv",
        model_path="models/hgb_model.joblib",
        target_column="target",
    ):
        self.train_path = train_path
        self.model_path = model_path
        self.target_column = target_column

    def load_data(self):
        data = pd.read_csv(self.train_path)

        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' was not found.")

        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        return X, y

    def build_model(self):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=500,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        )

    def validate_model(self, model, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        logger.info("Validation metrics:")
        logger.info(f"RMSE: {root_mean_squared_error(y_valid, predictions):.6f}")
        logger.info(f"MAE : {mean_absolute_error(y_valid, predictions):.6f}")
        logger.info(f"R2  : {r2_score(y_valid, predictions):.6f}")

    def save_model(self, model):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        logger.info(f"Model saved to: {self.model_path}")

    def run(self):
        X, y = self.load_data()

        #preprocessor = DataPreprocessor()
        #X = preprocessor.transform(X)
        validation_model = self.build_model()
        self.validate_model(validation_model, X, y)

        final_model = self.build_model()
        final_model.fit(X, y)

        self.save_model(final_model)


if __name__ == "__main__":
    trainer = RegressionTrainer()
    trainer.run()