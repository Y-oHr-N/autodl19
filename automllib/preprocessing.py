import collections

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit


class Clip(BaseEstimator, TransformerMixin):
    def __init__(self, low: float = 0.1, high: float = 99.9) -> None:
        self.low = low
        self.high = high

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'Clip':
        X = check_array(X, force_all_finite='allow-nan', estimator=self)

        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = check_array(X, force_all_finite='allow-nan', estimator=self)

        return np.clip(X, self.data_min_, self.data_max_)


class CountEncoder(BaseEstimator, TransformerMixin):
    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'CountEncoder':
        X = check_array(
            X,
            dtype=None,
            estimator=self,
            force_all_finite='allow-nan'
        )
        _, n_features = X.shape

        self.counters_ = [
            collections.Counter(X[:, j]) for j in range(n_features)
        ]

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = check_array(
            X,
            dtype=None,
            estimator=self,
            force_all_finite='allow-nan'
        )
        Xt = np.empty_like(X, dtype=float)
        _, n_features = X.shape

        vectorized = np.vectorize(
            lambda counter, xj: counter.get(xj, 0.0),
            excluded='counter'
        )

        for j in range(n_features):
            Xt[:, j] = vectorized(self.counters_[j], X[:, j])

        return Xt
