import logging

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit

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

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = pd.DataFrame(X)
        _, n_features = X.shape
        support = self.get_support()
        n_selected_features = np.sum(support)
        n_dropped_features = n_features - n_selected_features

        logger.info(
            f'{self.__class__.__name__} selects {n_selected_features} '
            f'features and drops {n_dropped_features} features.'
        )

        return X.iloc[:, support]


class DropDuplicates(BaseSelector, TransformerMixin):
    def __init__(self) -> None:
        pass

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropDuplicates':
        X = pd.DataFrame(X)

        self.duplicated_ = X.T.duplicated().values

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return ~self.duplicated_


class DropUniqueKey(BaseSelector, TransformerMixin):
    def __init__(self) -> None:
        pass

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropUniqueKey':
        X = pd.DataFrame(X)

        self.n_samples_ = len(X)
        self.nunique_ = X.nunique().values

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ != self.n_samples_


class NAProportionThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NAProportionThreshold':
        X = pd.DataFrame(X)
        n_samples = len(X)

        self.na_propotion_ = X.isnull().sum().values / n_samples

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.na_propotion_ < self.threshold


class NUniqueThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: int = 1) -> None:
        self.threshold = threshold

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NUniqueThreshold':
        X = pd.DataFrame(X)

        self.nunique_ = X.nunique().values

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ > self.threshold
