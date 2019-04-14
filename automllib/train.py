import logging

from typing import Any
from typing import Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

from hyperopt import fmin
from hyperopt import hp
from hyperopt import space_eval
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import check_scoring
from sklearn.model_selection import train_test_split

from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    gbdt = lgb.LGBMClassifier(
        learning_rate=0.01,
        max_depth=7,
        metric='auc',
        random_state=0
    )

    classifier = BalancedBaggingClassifier(gbdt)

    best_params = hyperopt_lightgbm(classifier, X, y)

    classifier.set_params(**best_params)

    classifier.fit(X, y)

    return classifier


@timeit
def hyperopt_lightgbm(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    scorer = check_scoring(estimator, scoring='roc_auc')

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        random_state=0,
        shuffle=False,
        test_size=0.5
    )

    space = {
        'base_estimator__num_leaves': hp.choice(
            'base_estimator__num_leaves', np.linspace(10, 200, 50, dtype=int)
        ),
        'base_estimator__feature_fraction': hp.quniform(
            'base_estimator__feature_fraction', 0.5, 1.0, 0.1
        ),
        'base_estimator__bagging_fraction': hp.quniform(
            'base_estimator__bagging_fraction', 0.5, 1.0, 0.1
        ),
        'base_estimator__bagging_freq': hp.choice(
            'base_estimator__bagging_freq', np.linspace(0, 50, 10, dtype=int)
        ),
        'base_estimator__reg_alpha': hp.uniform(
            'base_estimator__reg_alpha', 0, 2
        ),
        'base_estimator__reg_lambda': hp.uniform(
            'base_estimator__reg_lambda', 0, 2
        ),
        'base_estimator__min_child_weight': hp.uniform(
            'base_estimator__min_child_weight', 0.5, 10
        )
    }

    def objective(params):
        estimator.set_params(**params)

        estimator.fit(X_train, y_train)

        score = scorer(estimator, X_valid, y_valid)

        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        trials=trials,
        algo=tpe.suggest,
        max_evals=10,
        rstate=np.random.RandomState(1)
    )

    hyperparams = space_eval(space, best)

    logger.info(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

    return hyperparams
