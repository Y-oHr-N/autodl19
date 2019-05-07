import logging

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted

from .base import BaseTransformer
from .base import ONE_DIM_ARRAY_TYPE
from .base import TWO_DIM_ARRAY_TYPE
from .utils import timeit

logger = logging.getLogger(__name__)


class BaseSelector(BaseTransformer):
    @abstractmethod
    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        pass

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

        X = self._check_array(X)
        _, n_features = X.shape
        support = self.get_support()
        n_selected_features = len(support)
        n_dropped_features = n_features - n_selected_features

        logger.info(
            f'{self.__class__.__name__} selects {n_selected_features} '
            f'features and drops {n_dropped_features} features.'
        )

        return X[:, support]


class DropDuplicates(BaseSelector):
    def __init__(self, **params: Any) -> None:
        raise NotImplementedError()


class DropInvariant(BaseSelector):
    def __init__(self, **params: Any) -> None:
        pass

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, ['nunique_'])

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropInvariant':
        self._check_params()

        X = self._check_array(X)

        self.nunique_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ > 1


class DropUniqueKey(BaseSelector):
    def __init__(self, **params: Any) -> None:
        pass

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, ['nunique_'])

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropUniqueKey':
        self._check_params()

        X = self._check_array(X)

        self.n_samples_ = len(X)
        self.nunique_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ < self.n_samples_


class NAProportionThreshold(BaseSelector):
    def __init__(self, threshold: float = 0.6) -> None:
        pass

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, ['count_'])

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NAProportionThreshold':
        self._check_params()

        X = self._check_array(X)

        self.n_samples_ = len(X)

        self.count_ = np.array([pd.Series.count(column) for column in X.T])

        return self

    def get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.count_ >= (1.0 - self.threshold) * self.n_samples_
