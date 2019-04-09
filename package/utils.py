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

        start_time = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time

        logger.info(
            f'End [{func.__name__}]. Time elapsed: {elapsed_time:0.3f} sec.'
        )

        return res

    return timed


class Timer(object):
    def __init__(self):
        self.start = time.perf_counter()
        self.history = [self.start]

    def check(self, info):
        current = time.perf_counter()

        logger.info(f'[{info}] spend {current - self.history[-1]:0.3f} sec')

        self.history.append(current)


class Config(object):
    def __init__(self, info):
        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}

        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

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
        return self["time_budget"] - (time.time() - self["start_time"])

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
