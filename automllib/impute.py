from typing import Any
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseTransformer
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class SimpleImputer(BaseTransformer):
    _attributes = ['fill_value_']
    _validate = True

    def __init__(
        self,
        copy: bool = True,
        dtype: Union[str, Type] = None,
        fill_value: Any = None,
        strategy: str = 'mean',
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

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
        _, n_features = X.shape

        self.fill_value_ = np.empty(n_features, dtype=X.dtype)

        if self.strategy == 'constant':
            self.fill_value_[:] = self.fill_value
        elif self.strategy == 'min':
            for j, column in enumerate(X.T):
                is_nan = pd.isnull(column)
                self.fill_value_[j] = np.min(column[~is_nan])
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}.')

        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        if self.copy:
            X = np.copy(X)

        for j, column in enumerate(X.T):
            is_nan = pd.isnull(column)
            X[is_nan, j] = self.fill_value_[j]

        return X
