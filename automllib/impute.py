from typing import Any
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BasePreprocessor
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class SimpleImputer(BasePreprocessor):
    _attributes = ['statistics_']

    def __init__(
        self,
        copy: bool = True,
        dtype: Union[str, Type] = None,
        fill_value: Any = None,
        n_jobs: int = 1,
        strategy: str = 'mean',
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

        self.copy = copy
        self.fill_value = fill_value
        self.strategy = strategy

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'SimpleImputer':
        dtype = self.dtype

        if X.dtype.kind not in ('f', 'i', 'u'):
            dtype = X.dtype

        _, n_features = X.shape

        if self.strategy == 'constant':
            self.statistics_ = np.full(
                n_features,
                self.fill_value,
                dtype=dtype
            )

        elif self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0, dtype=dtype)

        elif self.strategy == 'min':
            self.statistics_ = np.empty(n_features, dtype=dtype)

            for j, column in enumerate(X.T):
                is_nan = pd.isnull(column)
                self.statistics_[j] = np.min(column[~is_nan])

        else:
            raise ValueError(f'Unknown strategy: {self.strategy}.')

        return self

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype

        if X.dtype.kind not in ('f', 'i', 'u'):
            dtype = X.dtype

        X = X.astype(dtype, copy=self.copy)

        for j, column in enumerate(X.T):
            is_nan = pd.isnull(column)
            X[is_nan, j] = self.statistics_[j]

        return X
