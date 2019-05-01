import logging
import time

from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np

from .constants import CATEGORICAL_PREFIX as C_PREFIX
from .constants import MULTI_VALUE_CATEGORICAL_PREFIX as M_PREFIX
from .constants import NUMERICAL_PREFIX as N_PREFIX
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TIME_PREFIX as T_PREFIX
from .constants import TWO_DIM_ARRAY_TYPE

logger = logging.getLogger(__name__)


def get_feature_names_by_prefix(
    X: TWO_DIM_ARRAY_TYPE,
    prefix: str
) -> ONE_DIM_ARRAY_TYPE:
    is_startwith = X.columns.str.startswith(prefix)
    n_features = is_startwith.sum()

    logger.info(f'Number of features starting with {prefix} is {n_features}.')

    return X.columns[is_startwith]


def get_categorical_feature_names(X: TWO_DIM_ARRAY_TYPE) -> ONE_DIM_ARRAY_TYPE:
    return get_feature_names_by_prefix(X, C_PREFIX)


def get_multi_value_categorical_feature_names(
    X: TWO_DIM_ARRAY_TYPE
) -> ONE_DIM_ARRAY_TYPE:
    return get_feature_names_by_prefix(X, M_PREFIX)


def get_numerical_feature_names(X: TWO_DIM_ARRAY_TYPE) -> ONE_DIM_ARRAY_TYPE:
    return get_feature_names_by_prefix(X, N_PREFIX)


def get_time_feature_names(X: TWO_DIM_ARRAY_TYPE) -> ONE_DIM_ARRAY_TYPE:
    return get_feature_names_by_prefix(X, T_PREFIX)


def timeit(func: Callable) -> Callable:
    def timed(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
        logger.info(f'==> Start {func.__name__}.')

        timer = Timer()
        ret = func(*args, **kwargs)
        elapsed_time = timer.get_elapsed_time()

        logger.info(f'==> End {func.__name__}. ({elapsed_time:.3f} sec.)')

        return ret

    return timed


class Timer(object):
    def __init__(self, time_budget: float = None) -> None:
        self.time_budget = time_budget

        self._start_time = time.perf_counter()

    def get_elapsed_time(self) -> float:
        return time.perf_counter() - self._start_time

    def get_remaining_time(self) -> float:
        if self.time_budget is None:
            raise RuntimeError('time_budget should be set.')

        remaining_time = self.time_budget - self.get_elapsed_time()

        return np.maximum(0.0, remaining_time)

    def check_remaining_time(self) -> None:
        if self.get_remaining_time() == 0.0:
            raise RuntimeError('Execution time limit has been exceeded.')
