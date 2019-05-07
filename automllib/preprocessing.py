import collections

from typing import Type
from typing import Union

import numpy as np

from sklearn.utils.validation import check_is_fitted

from .base import BaseTransformer
from .base import ONE_DIM_ARRAY_TYPE
from .base import TWO_DIM_ARRAY_TYPE
from .utils import timeit


class Clip(BaseTransformer):
    def __init__(
        self,
        dtype: Union[str, Type] = 'float64',
        low: float = 0.1,
        high: float = 99.9,
    ) -> None:
        self.dtype = dtype
        self.low = low
        self.high = high

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, ['data_max_', 'data_min_'])

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'Clip':
        self._check_params()

        X = self._check_array(X)

        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

        X = self._check_array(X)
        X = np.clip(X, self.data_min_, self.data_max_)

        return X.astype(self.dtype)


class CountEncoder(BaseTransformer):
    def __init__(self, dtype: Union[str, Type] = 'float64') -> None:
        self.dtype = dtype

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, ['counters_'])

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'CountEncoder':
        self._check_params()

        X = self._check_array(X)

        self.counters_ = [collections.Counter(column) for column in X.T]

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

        X = self._check_array(X)
        Xt = np.empty_like(X, dtype=self.dtype)
        vectorized = np.vectorize(
            lambda counter, xj: counter.get(xj, 0.0),
            excluded='counter'
        )

        for j, column in enumerate(X.T):
            Xt[:, j] = vectorized(self.counters_[j], column)

        return Xt
