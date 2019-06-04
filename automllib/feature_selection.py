from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd

from .base import BaseSelector
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class DropDuplicates(BaseSelector):
    pass


class DropCollinearFeatures(BaseSelector):
    _attributes = ['corr_']

    def __init__(self, threshold: float = 0.95, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'DropCollinearFeatures':
        X = X.astype('float64')

        self.corr_ = pd._libs.algos.nancorr(X)

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        triu = np.triu(self.corr_, k=1)
        triu = np.abs(triu)
        triu = np.nan_to_num(triu)

        return np.all(triu <= self.threshold, axis=0)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}


class FrequencyThreshold(BaseSelector):
    _attributes = ['frequency_', 'n_samples_']

    def __init__(
        self,
        max_frequency: Union[int, float] = 1.0,
        min_frequency: Union[int, float] = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.max_frequency = max_frequency
        self.min_frequency = min_frequency

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'FrequencyThreshold':
        self.n_samples_, _ = X.shape
        self.frequency_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        max_frequency = self.max_frequency
        min_frequency = self.min_frequency

        if isinstance(max_frequency, float):
            max_frequency = int(max_frequency * self.n_samples_)

        if isinstance(min_frequency, float):
            min_frequency = int(min_frequency * self.n_samples_)

        return (self.frequency_ > min_frequency) \
            & (self.frequency_ < max_frequency)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}


class NAProportionThreshold(BaseSelector):
    _attributes = ['count_', 'n_samples_']

    def __init__(self, threshold: float = 0.6, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'NAProportionThreshold':
        self.n_samples_, _ = X.shape
        self.count_ = np.array([pd.Series.count(column) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        return self.count_ >= (1.0 - self.threshold) * self.n_samples_

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}
