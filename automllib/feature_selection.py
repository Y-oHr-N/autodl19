from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils import safe_mask
from scipy.sparse import issparse

from scipy.stats import ks_2samp

from .base import BaseSelector
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


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


class DropDuplicates(BaseSelector):
    pass


class DropDriftFeatures(BaseSelector):
    _attributes = ['support_']

    def __init__(
        self,
        n_test: int = 5,
        n_test_samples: int = 100,
        random_state: Union[int, np.random.RandomState] = None,
        threshold: float = 0.05,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.n_test_samples = n_test_samples
        self.n_test = n_test
        self.random_state = random_state
        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        **kwargs: Any
    ) -> 'DropDriftFeatures':

        if not ('X_valid' in kwargs.keys()):
            self.support_ = None

            return self

        X_valid = kwargs['X_valid']
        random_state = check_random_state(self.random_state)

        self.support_ = np.full(self.n_features_, False)

        for test_idx in range(self.n_test):
            sample_indices1 = random_state.choice(np.arange(X.shape[0]), size=self.n_test_samples)
            sample_indices2 = random_state.choice(np.arange(X_valid.shape[0]), size=self.n_test_samples)

            if issparse(X):
                p_values = np.array([ks_2samp(np.squeeze(col1.toarray()), np.squeeze(col2.toarray()))[1]
                                     for col1, col2
                                     in zip(X[sample_indices1, :].T, X_valid[sample_indices2, :].T)]
                                    )
            else:
                p_values = np.array([ks_2samp(col1, col2)[1]
                                     for col1, col2
                                     in zip(X[sample_indices1, :].T, X_valid[sample_indices2, :].T)]
                                    )

            self.support_ += (p_values > self.threshold)

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        if self.support_ is None:
            return np.ones(self.n_features_, dtype=bool)

        return self.support_

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'X_types': ['2darray', 'sparse']}


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
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}


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
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}
