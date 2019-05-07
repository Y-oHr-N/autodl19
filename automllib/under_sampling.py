from typing import Any
from typing import Tuple
from typing import Union

import numpy as np

from imblearn.utils import check_sampling_strategy
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.utils.validation import check_is_fitted

from .base import BaseEstimator
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit


class RandomUnderSampler(BaseEstimator):
    _sampling_type = 'under-sampling'

    def __init__(
        self,
        random_state: Union[int, np.random.RandomState] = None,
        sampling_strategy: str = 'auto'
    ):
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(
            self,
            ['classes_', 'sample_indices_', 'sampling_strategy_']
        )

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE,
        **fit_params: Any
    ) -> 'RandomUnderSampler':
        random_state = check_random_state(self.random_state)
        n_samples = len(X)
        indices = np.arange(n_samples)
        arrays = []

        self.classes_ = np.unique(y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy,
            y,
            self._sampling_type
        )

        for target_class in self.classes_:
            sample_indices = indices[y == target_class]

            if target_class in self.sampling_strategy_:
                n_samples_per_class = self.sampling_strategy_[target_class]
                sample_indices = random_state.choice(
                    sample_indices,
                    size=n_samples_per_class,
                    replace=False
                )

            arrays.append(sample_indices)

        self.sample_indices_ = np.concatenate(arrays)

        self.sample_indices_.sort()

        return self

    def fit_resample(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE,
        **fit_params: Any
    ) -> Tuple[TWO_DIM_ARRAY_TYPE, ONE_DIM_ARRAY_TYPE]:
        self.fit(X, y, **fit_params)

        X = safe_indexing(X, self.sample_indices_)
        y = safe_indexing(y, self.sample_indices_)

        return X, y
