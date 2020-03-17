import logging
import time

from typing import Any
from typing import Callable
from typing import Tuple


class Timeit(object):
    def __init__(self, logger: logging.Logger = None) -> None:
        self.logger = logger

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args: Tuple, **kwargs: Any) -> Any:
            if self.logger is None:
                logger = logging.getLogger(__name__)
            else:
                logger = self.logger

            logger.info(f'==> Start {func}.')

            timer = Timer()

            timer.start()

            ret = func(*args, **kwargs)
            elapsed_time = timer.get_elapsed_time()

            logger.info(f'==> End {func}. ({elapsed_time:.3f} sec.)')

            return ret

        return wrapper


class Timer(object):
    def __init__(self, time_budget: float = None) -> None:
        self.time_budget = time_budget

    def get_elapsed_time(self) -> float:
        return time.perf_counter() - self.start_time_

    def get_remaining_time(self) -> float:
        if self.time_budget is None:
            raise RuntimeError('time_budget should be set.')

        remaining_time = self.time_budget - self.get_elapsed_time()

        return max(0.0, remaining_time)

    def check_remaining_time(self) -> None:
        if self.get_remaining_time() == 0.0:
            raise RuntimeError('Execution time limit has been exceeded.')

    def start(self) -> None:
        self.start_time_ = time.perf_counter()
