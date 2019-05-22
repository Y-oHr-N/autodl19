from typing import Any
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseTransformer
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .base import ONE_DIM_ARRAYLIKE_TYPE


class SimpleImputer(BaseTransformer):
    _attributes = []
    _validate = True

    def __init__(
        self,
        copy: bool = True,
        dtype: Union[str, Type] = None,
        fill_value: Any = None,
        strategy: str = 'mean',
        verbose: bool = False
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

        self.copy = copy
        self.fill_value = fill_value
        self.strategy = strategy

    def _check_params(self) -> None:
        pass

    def _fit(self, X, y=None):
        return self

    def _transform(self, X):
        if self.copy:
            X = np.copy(X)

        is_nan = pd.isnull(X)
        X[is_nan] = self.fill_value

        return X
