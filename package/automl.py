from collections import Counter
import logging

import hyperopt
from hyperopt import hp
from hyperopt import space_eval
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(X: pd.DataFrame, y: pd.Series):
    learning_rate = 0.01
    max_depth = 7
    metric = 'auc'
    n_estimators = 1000
    random_state = 0
    test_size = 0.25
    early_stopping_rounds = 10

    sampler = RandomUnderSampler(random_state=random_state)

    logger.info(f'Original dataset shape {Counter(y)}')

    X_res, y_res = sampler.fit_resample(X, y)

    logger.info(f'Resampled dataset shape {Counter(y_res)}')

    classifier = lgb.LGBMClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        metric=metric,
        n_estimators=n_estimators,
        random_state=random_state
    )

    best_params = hyperopt_lightgbm(classifier, X_res, y_res)

    classifier.set_params(**best_params)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_res,
        y_res,
        random_state=random_state,
        test_size=test_size
    )

    classifier.fit(
        X_train,
        y_train,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_valid, y_valid)]
    )

    return classifier


@timeit
def hyperopt_lightgbm(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
):
    metric = 'auc'
    random_state = 0
    test_size = 0.5
    early_stopping_rounds = 10

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        random_state=random_state,
        test_size=test_size
    )

    space = {
        'num_leaves': hp.choice(
            'num_leaves', np.linspace(10, 200, 50, dtype=int)
        ),
        'feature_fraction': hp.quniform('feature_fraction', 0.5, 1.0, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
        'bagging_freq': hp.choice(
            'bagging_freq', np.linspace(0, 50, 10, dtype=int)
        ),
        'reg_alpha': hp.uniform('reg_alpha', 0, 2),
        'reg_lambda': hp.uniform('reg_lambda', 0, 2),
        'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(params):
        estimator.set_params(**params)

        estimator.fit(
            X_train,
            y_train,
            early_stopping_rounds=early_stopping_rounds,
            eval_set=[(X_valid, y_valid)]
        )

        score = estimator.best_score_['valid_0'][metric]

        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=10, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)

    logger.info(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

    return hyperparams
