import logging

from typing import Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

from .objective import Objective
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(
    X: pd.DataFrame,
    y: pd.Series,
    error_score: Union[float, str] = np.nan,
    n_jobs: int = 1,
    n_trials: int = 10,
    random_state: Union[int, np.random.RandomState] = None,
    timeout: float = None
) -> lgb.LGBMClassifier:
    random_state = check_random_state(random_state)
    seed = random_state.randint(0, np.iinfo('int32').max)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler)
    classifier = lgb.LGBMClassifier(metric='auc', random_state=seed)
    objective = Objective(classifier, X, y, error_score=error_score)

    study.optimize(
        objective,
        n_jobs=n_jobs,
        n_trials=n_trials,
        timeout=timeout
    )

    logger.info(f'The best score is {-study.best_value:.3f}.')

    classifier.set_params(**study.best_params)
    classifier.fit(X, y)

    return classifier


@timeit
def resample(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: Union[int, float] = 100,
    random_state: Union[int, np.random.RandomState] = None,
):
    logger.info(f'The shape of X before under-sampling is {X.shape}')

    if type(max_samples) is float:
        max_samples = int(max_samples * len(X))

    # classes = np.unique(y)
    # n_classes = len(classes)
    # max_samples_per_class = max_samples // n_classes

    resampler = RandomUnderSampler(
        random_state=random_state,
        # sampling_strategy=sampling_strategy
    )

    resampler.fit_resample(X, y)

    logger.info(f'The shape of X after under-sampling is {X.shape}')

    X = safe_indexing(X, resampler.sample_indices_)
    y = safe_indexing(y, resampler.sample_indices_)

    return X, y
