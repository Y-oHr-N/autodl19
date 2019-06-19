import collections
import itertools

from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BasePreprocessor
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class Clip(BasePreprocessor):
    """Clip (limit) the values in an array.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.preprocessing import Clip
    >>> pre = Clip()
    >>> X = [[10, np.nan, 4], [0, 2, 1]]
    >>> pre.fit_transform(X)
    array([[9.9 ,  nan, 3.97],
           [0.1 , 2.  , 1.03]])
    """

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        high: float = 99.0,
        low: float = 1.0,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

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

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        return np.clip(X, self.data_min_, self.data_max_)


class CountEncoder(BasePreprocessor):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.preprocessing import CountEncoder
    >>> pre = CountEncoder()
    >>> X = [[1, 1, 'Cat'], [2, 2, np.nan], [1, 1, np.nan], [1, 1, np.nan]]
    >>> pre.fit_transform(X)
    array([[3., 3., 1.],
           [1., 1., 3.],
           [3., 3., 3.],
           [3., 3., 3.]])
    """

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'CountEncoder':
        self.counters_ = [collections.Counter(column) for column in X.T]

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype
        n_samples, n_features = X.shape
        Xt = np.empty((n_samples, n_features), dtype=dtype)
        vectorized = np.vectorize(
            lambda counter, xj: counter.get(xj, 0.0),
            excluded='counter'
        )

        for j, column in enumerate(X.T):
            Xt[:, j] = vectorized(self.counters_[j], column)

        return Xt


class ModifiedStandardScaler(BasePreprocessor):
    """Standardize features by removing the mean and scaling to unit variance.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.preprocessing import ModifiedStandardScaler
    >>> pre = ModifiedStandardScaler()
    >>> X = [[0, np.nan], [0, 0], [1, np.nan], [1, 1]]
    >>> pre.fit_transform(X)
    array([[-1., nan],
           [-1., -1.],
           [ 1., nan],
           [ 1.,  1.]])
    """

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'ModifiedStandardScaler':
        self.mean_ = np.nanmean(X, axis=0, dtype=self.dtype)
        self.std_ = np.nanstd(X, axis=0, dtype=self.dtype)
        self.scale_ = self.std_.copy()
        self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        return (X - self.mean_) / self.scale_


class RowStatistics(BasePreprocessor):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.preprocessing import RowStatistics
    >>> pre = RowStatistics()
    >>> X = [[0, np.nan], [0, 0], [1, np.nan], [1, 1]]
    >>> pre.fit_transform(X)
    array([[1.],
           [0.],
           [1.],
           [0.]])
    """

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'RowStatistics':
        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {
            'allow_nan': True,
            'stateless': True,
            'X_types': ['2darray', 'str']
        }

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype

        if dtype is None:
            if X.dtype.kind in ('f', 'i', 'u'):
                dtype = X.dtype
            else:
                dtype = 'float64'

        is_nan = pd.isnull(X)

        return np.sum(is_nan, axis=1, dtype=dtype).reshape(-1, 1)


class ArithmeticalFeatures(BasePreprocessor):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.preprocessing import ArithmeticalFeatures
    >>> pre = ArithmeticalFeatures()
    >>> X = [[1, 1, 100], [2, 2, 10], [1, 1, 1], [1, 1, np.nan]]
    >>> pre.fit_transform(X)
    array([[  0., -99., -99.],
           [  0.,  -8.,  -8.],
           [  0.,   0.,   0.],
           [  0.,  nan,  nan]])
    """

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        operand: str = 'subtract',
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

        self.operand = operand

    def _check_params(self) -> None:
        if self.operand not in ['add', 'subtract', 'multiply', 'divide']:
            raise ValueError(f'Unknown operand: {self.operand}')

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'ArithmeticalFeatures':
        self.n_output_features_ = \
            self.n_features_ * (self.n_features_ - 1) // 2

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'stateless': True}

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype

        if dtype is None:
            if X.dtype.kind in ('f', 'i', 'u'):
                dtype = X.dtype

        n_samples, _ = X.shape
        Xt = np.empty((n_samples, self.n_output_features_), dtype=dtype)
        iterable = itertools.combinations(range(self.n_features_), 2)
        func = getattr(np, self.operand)

        for j, (k, l) in enumerate(iterable):
            Xt[:, j] = func(X[:, k], X[:, l])

        return Xt


class TextStatistics(BasePreprocessor):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.preprocessing import TextStatistics
    >>> pre = TextStatistics()
    >>> X = [['Cat'], ['Cow'], ['Mouse'], ['Lion']]
    >>> pre.fit_transform(X)
    array([[3.],
           [3.],
           [5.],
           [4.]])
    """

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'TextStatistics':
        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'X_types': ['2darray', 'str']}

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype
        n_samples, n_features = X.shape
        Xt = np.empty((n_samples, n_features), dtype=dtype)
        vectorized = np.vectorize(len)

        for j, column in enumerate(X.T):
            Xt[:, j] = vectorized(column)

        return Xt
