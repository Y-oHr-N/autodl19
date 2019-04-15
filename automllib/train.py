import logging

import lightgbm as lgb
import optuna
import pandas as pd

from imblearn.ensemble import BalancedBaggingClassifier

from .objective import Objective
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    gbdt = lgb.LGBMClassifier(
        learning_rate=0.01,
        max_depth=7,
        metric='auc',
        random_state=0,
        subsample_freq=1
    )

    clf = BalancedBaggingClassifier(
        base_estimator=gbdt,
        max_samples=1_000,
        random_state=0
    )

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    objective = Objective(clf, X, y, cv=3)

    study.optimize(objective, n_trials=10)

    logger.info(f'The AUC is {-study.best_value:.3f}.')

    clf.set_params(**study.best_params)

    clf.fit(X, y)

    return clf
