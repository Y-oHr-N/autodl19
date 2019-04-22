import logging

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)


class BaseSelector(BaseEstimator):
    def transform(self, X: np.ndarray) -> np.ndarray:
        support = self.get_support()

        logger.info(f'{support.sum()} features are selected.')

        return X.loc[:, support]


class NAProportionThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> 'NAProportionThreshold':
        n_samples = len(X)

        self.na_propotions_ = X.isnull().sum() / n_samples

        return self

    def get_support(self) -> np.ndarray:
        return self.na_propotions_ < self.threshold


class NUniqueThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> 'NUniqueThreshold':
        self.nunique_ = X.nunique()

        return self

    def get_support(self) -> np.ndarray:
        return self.nunique_ > self.threshold
