import logging

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)


class Clip(BaseEstimator, TransformerMixin):
    def __init__(self, low: float = 0.1, high: float = 99.9) -> None:
        self.low = low
        self.high = high

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Clip':
        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if hasattr(X, 'values'):
            return X.clip(self.data_min_, self.data_max_, axis=1)
        else:
            return X.clip(self.data_min_, self.data_max_)
