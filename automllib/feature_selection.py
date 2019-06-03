from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from scipy.stats import ks_2samp

from .base import BaseSelector
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class DropDuplicates(BaseSelector):
    pass


class DropCollinearFeatures(BaseSelector):
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


class DropDriftFeatures(BaseSelector):
    def __init__(
        self,
        threshold: float = 0.1,
        verbose: int = 0,
        n_test_samples: int = 100,
        n_test: int = 5,
        random_state: int = 2019
    ) -> None:
        super().__init__(verbose=verbose)

        self.threshold = threshold
        # self.p_values_array = None
        self.support = None
        self.n_test_samples = n_test_samples   # the num. of samples for KS-test
        self.n_test = n_test                   # the num. of KS-test
        self.random_state = random_state       #

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        **kwargs: Any
    ) -> 'DropDriftFeatures':

        if not ('X_valid' in kwargs.keys()):
            raise ValueError
        X_valid = kwargs['X_valid']

        random_state = check_random_state(self.random_state)

        n_samples, n_dims = X.shape

        self.support = np.full(n_dims, False)
        sampled_X = np.zeros((self.n_test_samples, n_dims))
        sampled_X_valid = np.zeros((self.n_test_samples, n_dims))

        for test_idx in range(self.n_test):
            for dim_idx in range(n_dims):
                sampled_X[:, dim_idx] = random_state.choice(X[:, dim_idx],
                                                            size=self.n_test_samples)
                sampled_X_valid[:, dim_idx] = random_state.choice(X_valid[:, dim_idx],
                                                                  size=self.n_test_samples)

            p_values = np.array([ks_2samp(col1, col2)[1]
                                 for col1, col2
                                 in zip(sampled_X.T, sampled_X_valid.T)]
                                )
            self.support += (p_values > self.threshold)

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        return self.support
