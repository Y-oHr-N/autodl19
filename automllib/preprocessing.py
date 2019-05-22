import collections
import itertools

from typing import List
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from joblib import delayed
from joblib import effective_n_jobs
from joblib import Parallel
from sklearn.utils import gen_even_slices
from sklearn.utils import safe_indexing

from .base import BaseTransformer
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


def count_encode(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    counters: List[collections.Counter],
) -> TWO_DIM_ARRAYLIKE_TYPE:
    Xt = np.empty_like(X)
    vectorized = np.vectorize(
        lambda counter, xj: counter.get(xj, 0.0),
        excluded='counter'
    )

    for j, column in enumerate(X.T):
        Xt[:, j] = vectorized(counters[j], column)

    return Xt


def subtracted_features(X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
    n_samples, n_input_features = X.shape
    n_output_features = n_input_features * (n_input_features - 1) // 2
    Xt = np.empty((n_samples, n_output_features), dtype=X.dtype)
    iterable = itertools.combinations(range(n_input_features), 2)

    for j, (k, l) in enumerate(iterable):
        Xt[:, j] = X[:, k] - X[:, l]

    return Xt


class Clip(BaseTransformer):
    _attributes = ['data_max_', 'data_min_']
    _validate = True

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        high: float = 99.9,
        low: float = 0.1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

        self.high = high
        self.low = low

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'Clip':
        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return np.clip(X, self.data_min_, self.data_max_)


class CountEncoder(BaseTransformer):
    _attributes = ['counters_']
    _validate = True

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

        self.n_jobs = n_jobs

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'CountEncoder':
        self.counters_ = [collections.Counter(column) for column in X.T]

        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        n_samples, _ = X.shape
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(count_encode)
        result = parallel(
            func(
                safe_indexing(X, s), self.counters_
            ) for s in gen_even_slices(n_samples, n_jobs)
        )

        return np.concatenate(result)


class SubtractedFeatures(BaseTransformer):
    _attributes = []
    _validate = True

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

        self.n_jobs = n_jobs

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'SubtractedFeatures':
        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        n_samples, _ = X.shape
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(subtracted_features)
        result = parallel(
            func(
                safe_indexing(X, s)
            ) for s in gen_even_slices(n_samples, n_jobs)
        )

        return np.concatenate(result)


class RowStatistics(BaseTransformer):
    _attributes = []
    _validate = True

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'RowStatistics':
        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        return pd.isnull(X).sum(axis=1).reshape(-1, 1)


class StandardScaler(BaseTransformer):
    _attributes = ['mean_', 'std_', 'scale_']
    _validate = True

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'StandardScaler':
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.scale_ = self.std_.copy()
        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        return (X - self.mean_) / self.scale_
