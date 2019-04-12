import logging
import time

from .constants import AGGREGATE_FUNCTIONS_MAP
from .constants import CATEGORICAL_PREFIX
from .constants import CATEGORICAL_TYPE
from .constants import MULTI_VALUE_CATEGORICAL_PREFIX
from .constants import MULTI_VALUE_CATEGORICAL_TYPE
from .constants import NUMERICAL_PREFIX
from .constants import NUMERICAL_TYPE
from .constants import TIME_PREFIX

logger = logging.getLogger(__name__)


def timeit(func):
    def timed(*args, **kwargs):
        logger.info(f'Start [{func.__name__}].')

        timer = Timer()
        ret = func(*args, **kwargs)

        logger.info(
            f'End [{func.__name__}]. '
            f'The elapsed time is {timer.elapsed_time():0.3f} seconds.'
        )

        return ret

    return timed


class Timer(object):
    def __init__(self, time_budget: float = None) -> None:
        self.time_budget = time_budget

        self._start_time = time.perf_counter()

    def elapsed_time(self) -> float:
        return time.perf_counter() - self._start_time

    def remaining_time(self) -> float:
        if self.time_budget is None:
            raise RuntimeError('`time_budget` should be set.')

        ret = self.time_budget - self.elapsed_time()

        if ret <= 0.0:
            raise TimeoutError('The time limit has been exceeded.')

        return ret


class Config(object):
    def __init__(self, info):
        self.data = info.copy()
        self.data['tables'] = {}

        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        if col.startswith(NUMERICAL_PREFIX):
            return AGGREGATE_FUNCTIONS_MAP[NUMERICAL_TYPE]
        if col.startswith(CATEGORICAL_PREFIX):
            return AGGREGATE_FUNCTIONS_MAP[CATEGORICAL_TYPE]
        if col.startswith(MULTI_VALUE_CATEGORICAL_PREFIX):
            assert False, f"MultiCategory type feature's aggregate op are not supported."
            return AGGREGATE_FUNCTIONS_MAP[MULTI_VALUE_CATEGORICAL_TYPE]
        if col.startswith(TIME_PREFIX):
            assert False, f"Time type feature's aggregate op are not implemented."

        assert False, f"Unknown col type {col}"

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data
