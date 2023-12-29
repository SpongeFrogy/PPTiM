from typing import Literal
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np


class NoNormalizerError(Exception):
    "Only minmax and norm"


class PreprocessingError(Exception):
    "Transformed DataFrame has nan values"


class PreprocessingModel:

    @classmethod
    def do_drop(cls, col: pd.Series, p: float = 0.05) -> bool:
        """return True if col has more than p proportion of nan values and False if not

        Args:
            col (pd.Series): column of DataFrame
            p (float, optional): cut off proportion. Defaults to 0.05.

        Returns:
            bool:  True if col has more than p proportion of nan values and False if not
        """

        tf, counts = np.unique(col.isna(), return_counts=True)
        if len(counts) == 1:
            if tf[0] == True:
                return True
            else:
                return False
        f_pos = 0 if tf[0] == False else 1
        if counts[(f_pos + 1) % 2]/counts[f_pos] > p:
            return True
        else:
            return False

    @classmethod
    def check_transforms(cls, transformed_x: pd.DataFrame) -> None:
        """check if transformed_x has Nan 

        Args:
            transformed_x (pd.DataFrame): df to be checked

        Raises:
            PreprocessingError: if transformed_x has Nan 
        """

        if not np.all((transformed_x == transformed_x.dropna()).values):
            raise PreprocessingError("wrong fillna")

    def __init__(self, p_drop: float = 0.05, threshold: float = 0.0001, normalizer: Literal["minmax", "normalizer"] = "normalizer") -> None:
        """Preprocessing model for dataset:
        - drop columns with more than p_drop*100 % Nan values
        - VarianceThreshold(threshold=threshold)
        - normalize dataset

        Args:
            p_drop (float, optional): percentage of Nan values . Defaults to 0.05.
            threshold (float, optional): VarianceThreshold.threshold . Defaults to 0.0001.
            normalizer (Literal["minmax", "normalizer"], optional): kind of normalizer. Defaults to "normalizer".

        Raises:
            NoNormalizerError: if `normalizer` not implemented
        """
        self.p_drop = p_drop
        self.threshold = threshold

        if normalizer not in ("minmax", "normalizer"):
            raise NoNormalizerError("only minmax and norm")

        self.normalizer = Normalizer() if normalizer == "normalizer" else MinMaxScaler()

    def fit_transform(self, x_train: pd.DataFrame) -> pd.DataFrame:
        """fit preprocessing model and transform train DataFrame with drop and filling missing and normalizer

        Args:
            x_train (pd.DataFrame): x for fit and transform.
        """

        # drop cols with p_drop*100 % less of missing values
        transformed = x_train[[name for name in x_train.columns if not self.do_drop(
            x_train[name], self.p_drop)]]

        # fill missing values with mean of column (axis=1)
        self.means = transformed.mean()
        transformed = transformed.fillna(self.means).dropna(axis=1)

        self.check_transforms(transformed)  # check if has nan values

        # normalizer
        transformed = pd.DataFrame(self.normalizer.fit_transform(
            transformed), columns=transformed.columns, index=transformed.index)
        self.fit_cols = transformed.columns

        # drop columns with varian less than threshold
        self.selector = VarianceThreshold(threshold=self.threshold)
        self.selector.fit(transformed)
        # [True, False, ...] with len = len(x_train)
        kept = self.selector.get_support()
        # kept columns after VarianceThreshold
        self.cols = transformed.columns[kept]
        transformed = transformed.loc[:, self.cols]
        self.index = transformed.index

        return pd.DataFrame(transformed, columns=self.cols, index=self.index)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """transform x and y 

        Args:
            x (pd.DataFrame): x to transform 

        Returns:
            pd.DataFrame: transformed x.
        """

        transformed_x = x[self.fit_cols].fillna(self.means).dropna(axis=1)

        self.check_transforms(transformed_x)

        return pd.DataFrame(self.normalizer.transform(transformed_x), columns=self.fit_cols, index=x.index)[self.cols]


if __name__ == "__main__":
    df = pd.DataFrame([[3, 2], [1, np.nan]])
    model = PreprocessingModel()
    model.fit_transform(df)
