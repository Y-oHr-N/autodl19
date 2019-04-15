import collections
import logging

import lightgbm as lgb
import optuna
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from .objective import Objective
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    logger.info(f'{collections.Counter(y)}')

    rus = RandomUnderSampler(random_state=0)
    X, y = rus.fit_resample(X, y)

    logger.info(f'{collections.Counter(y)}')

    gbdt = lgb.LGBMClassifier(metric='auc', random_state=0)

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    objective = Objective(gbdt, X, y, error_score='raise')

    study.optimize(objective, n_jobs=-1, n_trials=10)

    logger.info(f'CV score is {-study.best_value:.3f}.')

    gbdt.set_params(**study.best_params)

    gbdt.fit(X, y)

    return gbdt
