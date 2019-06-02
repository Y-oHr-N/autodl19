from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from imblearn.utils import check_sampling_strategy
from sklearn.utils import check_random_state

from .base import BaseSampler
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class RandomUnderSampler(BaseSampler):
    _sampling_type = 'under-sampling'

    def __init__(
        self,
        random_state: Union[int, np.random.RandomState] = None,
        replacement: bool = False,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)

        self.random_state = random_state
        self.replacement = replacement
        self.sampling_strategy = sampling_strategy
        self.shuffle = shuffle

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        **fit_params: Any
    ) -> 'RandomUnderSampler':
        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape
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
                    replace=self.replacement
                )

            arrays.append(sample_indices)

        self.sample_indices_ = np.concatenate(arrays)

        if not self.shuffle:
            self.sample_indices_.sort()

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}
