from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseSelector


class DropDuplicates(BaseSelector):
    pass


class DropCollinearFeatures(BaseSelector):
    _attributes = ['corr_']

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        threshold: float = 0.95,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> 'DropCollinearFeatures':
        self.corr_ = X.corr().values

        return self

    def _get_support(self) -> np.ndarray:
        triu = np.triu(self.corr_, k=1)
        triu = np.abs(triu)
        triu = np.nan_to_num(triu)

        return np.all(triu <= self.threshold, axis=0)


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

    def _fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DropInvariant':
        self.nunique_ = X.nunique().values

        return self

    def _get_support(self) -> np.ndarray:
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

    def _fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DropUniqueKey':
        self.n_samples_, _ = X.shape
        self.nunique_ = X.nunique().values

        return self

    def _get_support(self) -> np.ndarray:
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
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> 'NAProportionThreshold':
        self.n_samples_, _ = X.shape
        self.count_ = X.count().values

        return self

    def _get_support(self) -> np.ndarray:
        return self.count_ >= (1.0 - self.threshold) * self.n_samples_
