import logging
import time

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
        self.data = {**info}
        self.data['tables'] = {}

        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

        self._timer = Timer(info['time_budget'])

    @staticmethod
    def aggregate_op(col):

        def my_nunique(x):
            return x.nunique()

        my_nunique.__name__ = 'nunique'
        ops = {
            NUMERICAL_TYPE: ["mean", "sum"],
            CATEGORICAL_TYPE: ["count"],
            #  TIME_TYPE: ["max"],
            #  MULTI_VALUE_CATEGORICAL_TYPE: [my_unique]
        }

        if col.startswith(NUMERICAL_PREFIX):
            return ops[NUMERICAL_TYPE]
        if col.startswith(CATEGORICAL_PREFIX):
            return ops[CATEGORICAL_TYPE]
        if col.startswith(MULTI_VALUE_CATEGORICAL_PREFIX):
            assert False, f"MultiCategory type feature's aggregate op are not supported."
            return ops[MULTI_VALUE_CATEGORICAL_TYPE]
        if col.startswith(TIME_PREFIX):
            assert False, f"Time type feature's aggregate op are not implemented."

        assert False, f"Unknown col type {col}"

    def time_left(self):
        return self._timer.remaining_time()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)
