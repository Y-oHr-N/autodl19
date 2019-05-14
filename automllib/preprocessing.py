import collections

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
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE


def count_encode(
    X: TWO_DIM_ARRAY_TYPE,
    counters: List[collections.Counter],
) -> TWO_DIM_ARRAY_TYPE:
    Xt = np.empty_like(X)
    vectorized = np.vectorize(
        lambda counter, xj: counter.get(xj, 0.0),
        excluded='counter'
    )

    for j, column in enumerate(X.T):
        Xt[:, j] = vectorized(counters[j], column)

    return Xt


class Clip(BaseTransformer):
    _attributes = ['data_max_', 'data_min_']

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        low: float = 0.1,
        high: float = 99.9,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

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
        return np.clip(X, self.data_min_, self.data_max_)


class CountEncoder(BaseTransformer):
    _attributes = ['counters_']

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
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'CountEncoder':
        self.counters_ = [collections.Counter(column) for column in X.T]

        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
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


class RowStatistics(BaseTransformer):
    _attributes = []

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
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'RowStatistics':
        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        n_samples, _ = X.shape
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lambda X: pd.isnull(X).sum(axis=1).reshape(-1, 1))
        result = parallel(
            func(
                safe_indexing(X, s)
            ) for s in gen_even_slices(n_samples, n_jobs)
        )

        return np.concatenate(result)



class StandardScaler(BaseTransformer):
    _attributes = ['mean_', 'std_']

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
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'StandardScaler':
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.scale_ = self.std_.copy()
        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        n_samples, _ = X.shape
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lambda X, mean, scale: (X - mean) / scale)
        result = parallel(
            func(
                safe_indexing(X, s), self.mean_, self.scale_
            ) for s in gen_even_slices(n_samples, n_jobs)
        )

        return np.concatenate(result)
