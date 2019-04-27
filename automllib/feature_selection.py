import logging

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE

logger = logging.getLogger(__name__)


class BaseSelector(BaseEstimator, ABC):
    @abstractmethod
    def __init__(self, **params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'BaseSelector':
        pass

    @abstractmethod
    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        pass

    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        _, n_features = X.shape
        support = self.get_support()
        n_selected_features = np.sum(support)
        n_dropped_features = n_features - n_selected_features

        logger.info(
            f'{n_selected_features} features are selected and '
            f'{n_dropped_features} features are dropped.'
        )

        return X.loc[:, support]


class DropUniqueKey(BaseSelector, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropUniqueKey':
        self.n_samples_ = len(X)
        self.nunique_ = X.nunique()

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ != self.n_samples_


class NAProportionThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NAProportionThreshold':
        n_samples = len(X)

        self.na_propotion_ = X.isnull().sum() / n_samples

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.na_propotion_ < self.threshold


class NUniqueThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NUniqueThreshold':
        self.nunique_ = X.nunique()

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ > self.threshold
