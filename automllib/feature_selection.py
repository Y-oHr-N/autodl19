import logging

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)


class BaseSelector(BaseEstimator, ABC):
    @abstractmethod
    def __init__(self, **params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> 'BaseSelector':
        pass

    @abstractmethod
    def get_support(self) -> np.ndarray:
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        _, n_features = X.shape
        support = self.get_support()
        n_selected_features = np.sum(support)

        logger.info(
            f'{n_selected_features} features are selected and '
            f'{n_features - n_selected_features} features are dropped.'
        )

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

        self.na_propotion_ = X.isnull().sum() / n_samples

        return self

    def get_support(self) -> np.ndarray:
        return self.na_propotion_ < self.threshold


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
