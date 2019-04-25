import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array


class Clip(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        copy: bool = True,
        low: float = 0.1,
        high: float = 99.9
    ) -> None:
        self.copy = copy
        self.low = low
        self.high = high

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Clip':
        X = check_array(X)

        self.data_min_, self.data_max_ = np.percentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)

        if self.copy:
            out = None
        else:
            out = X

        return np.clip(X, self.data_min_, self.data_max_, out=out)
