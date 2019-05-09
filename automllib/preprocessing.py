import collections

from typing import Type
from typing import Union

import numpy as np

from .base import BaseTransformer
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE


class Clip(BaseTransformer):
    _attributes = ['data_max_', 'data_min_']

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

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'Clip':
        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = np.clip(X, self.data_min_, self.data_max_)

        return X.astype(self.dtype)


class CountEncoder(BaseTransformer):
    _attributes = ['counters_']

    def __init__(self, dtype: Union[str, Type] = 'float64') -> None:
        self.dtype = dtype

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'CountEncoder':
        self.counters_ = [collections.Counter(column) for column in X.T]

        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        Xt = np.empty_like(X, dtype=self.dtype)
        vectorized = np.vectorize(
            lambda counter, xj: counter.get(xj, 0.0),
            excluded='counter'
        )

        for j, column in enumerate(X.T):
            Xt[:, j] = vectorized(self.counters_[j], column)

        return Xt
