import pandas as pd


class DataPreprocessor:
    """
    Class responsible for feature engineering and data preparation.
    """

    def __init__(self):
        # Features to remove
        self.columns_to_drop = [
            "6",
        ]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix.
        """

        X = X.copy()

        existing_columns = [
            col
            for col in self.columns_to_drop
            if col in X.columns
        ]

        X = X.drop(columns=existing_columns)

        return X