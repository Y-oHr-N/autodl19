import logging
import time

from typing import Callable
from typing import Dict

import numpy as np
import pandas as pd

from .constants import AGGREGATE_FUNCTIONS_MAP as AFS_MAP
from .constants import CATEGORICAL_PREFIX as C_PREFIX
from .constants import CATEGORICAL_TYPE as C_TYPE
from .constants import MULTI_VALUE_CATEGORICAL_PREFIX as M_PREFIX
from .constants import MULTI_VALUE_CATEGORICAL_TYPE as M_TYPE
from .constants import NUMERICAL_PREFIX as N_PREFIX
from .constants import NUMERICAL_TYPE as N_TYPE
from .constants import TIME_PREFIX as T_PREFIX
from .constants import TIME_TYPE as T_TYPE

logger = logging.getLogger(__name__)


def aggregate_functions(columns: pd.Index) -> Dict[str, Callable]:
    func = {}

    c_feature_names = columns[columns.str.startswith(C_PREFIX)]
    m_feature_names = columns[columns.str.startswith(M_PREFIX)]
    n_feature_names = columns[columns.str.startswith(N_PREFIX)]
    t_feature_names = columns[columns.str.startswith(T_PREFIX)]

    func.update({name: AFS_MAP[C_TYPE] for name in c_feature_names})
    func.update({name: AFS_MAP[M_TYPE] for name in m_feature_names})
    func.update({name: AFS_MAP[N_TYPE] for name in n_feature_names})
    func.update({name: AFS_MAP[T_TYPE] for name in t_feature_names})

    return func


def timeit(func):
    def timed(*args, **kwargs):
        logger.info(f'==== Start {func}. ====')

        timer = Timer()
        ret = func(*args, **kwargs)
        elapsed_time = timer.get_elapsed_time()

        logger.info(f'==== End {func}. ({elapsed_time:.3f} sec.) ====')

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


class Config(object):
    def __init__(self, info):
        self.data = info.copy()
        self.data['tables'] = {}

        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data
