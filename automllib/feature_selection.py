from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseSelector
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE


class DropDuplicates(BaseSelector):
    pass


class DropInvariant(BaseSelector):
    _attributes = ['nunique_']

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
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropInvariant':
        self.nunique_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ > 1


class DropUniqueKey(BaseSelector):
    _attributes = ['nunique_', 'n_samples_']

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
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropUniqueKey':
        self.n_samples_, _ = X.shape
        self.nunique_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.nunique_ < self.n_samples_


class NAProportionThreshold(BaseSelector):
    _attributes = ['count_', 'n_samples_']

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        threshold: float = 0.6,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NAProportionThreshold':
        self.n_samples_, _ = X.shape
        self.count_ = np.array([pd.Series.count(column) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAY_TYPE:
        return self.count_ >= (1.0 - self.threshold) * self.n_samples_
