import logging
import time

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from .constants import AGGREGATE_FUNCTIONS_MAP as AFS_MAP
from .constants import CATEGORICAL_PREFIX as C_PREFIX
from .constants import CATEGORICAL_TYPE as C_TYPE
from .constants import MULTI_VALUE_CATEGORICAL_PREFIX as M_PREFIX
# from .constants import MULTI_VALUE_CATEGORICAL_TYPE as M_TYPE
from .constants import NUMERICAL_PREFIX as N_PREFIX
from .constants import NUMERICAL_TYPE as N_TYPE
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TIME_PREFIX as T_PREFIX
# from .constants import TIME_TYPE as T_TYPE
from .constants import TWO_DIM_ARRAY_TYPE

logger = logging.getLogger(__name__)


def aggregate_functions(
    X: TWO_DIM_ARRAY_TYPE
) -> Dict[str, List[Union[str, Callable]]]:
    func = {}

    c_feature_names = get_categorical_columns(X)
    # m_feature_names = get_multi_value_categorical_columns(X)
    n_feature_names = get_numerical_columns(X)
    # t_feature_names = get_time_columns(X)

    func.update({name: AFS_MAP[C_TYPE] for name in c_feature_names})
    # func.update({name: AFS_MAP[M_TYPE] for name in m_feature_names})
    func.update({name: AFS_MAP[N_TYPE] for name in n_feature_names})
    # func.update({name: AFS_MAP[T_TYPE] for name in t_feature_names})

    return func


def get_columns_by_prefix(
    X: TWO_DIM_ARRAY_TYPE,
    prefix: str
) -> ONE_DIM_ARRAY_TYPE:
    columns = X.columns
    is_startwith = columns.str.startswith(prefix)
    n_features = is_startwith.sum()

    logger.info(f'Number of features starting with {prefix} is {n_features}.')

    return list(columns[is_startwith])


def get_categorical_columns(X: TWO_DIM_ARRAY_TYPE) -> ONE_DIM_ARRAY_TYPE:
    return get_columns_by_prefix(X, C_PREFIX)


def get_multi_value_categorical_columns(
    X: TWO_DIM_ARRAY_TYPE
) -> ONE_DIM_ARRAY_TYPE:
    return get_columns_by_prefix(X, M_PREFIX)


def get_numerical_columns(X: TWO_DIM_ARRAY_TYPE) -> ONE_DIM_ARRAY_TYPE:
    return get_columns_by_prefix(X, N_PREFIX)


def get_time_columns(X: TWO_DIM_ARRAY_TYPE) -> ONE_DIM_ARRAY_TYPE:
    return get_columns_by_prefix(X, T_PREFIX)


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
            raise RuntimeError('`time_budget` should be set.')

        remaining_time = self.time_budget - self.get_elapsed_time()

        return np.maximum(0.0, remaining_time)

    def check_remaining_time(self) -> None:
        if self.get_remaining_time() == 0.0:
            raise RuntimeError('Execution time limit has been exceeded.')
