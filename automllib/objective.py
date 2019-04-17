from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import cross_val_score


class Objective(object):
    def __init__(
        self,
        estimator: lgb.LGBMModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[int, BaseCrossValidator] = 5,
        error_score: Union[str, float] = np.nan,
        groups: pd.Series = None,
        scoring: Union[str, Callable] = None,
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv
        self.error_score = error_score
        self.groups = groups
        self.scoring = scoring

    def _get_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        params = {'n_jobs': 1}

        params['boosting_type'] = trial.suggest_categorical(
            'boosting_type',
            choices=[
                # 'dart',
                'gbdt',
                # 'goss'
            ]
        )

        params['colsample_bytree'] = trial.suggest_uniform(
            'colsample_bytree',
            low=0.5,
            high=1.0
        )

        params['learning_rate'] = trial.suggest_loguniform(
            'learning_rate',
            low=0.001,
            high=0.1
        )

        params['num_leaves'] = trial.suggest_int(
            'num_leaves',
            low=2,
            high=123
        )

        params['reg_alpha'] = trial.suggest_loguniform(
            'reg_alpha',
            low=1e-06,
            high=10.0
        )

        params['reg_lambda'] = trial.suggest_loguniform(
            'reg_lambda',
            low=1e-06,
            high=10.0
        )

        if params['boosting_type'] == 'goss':
            params['other_rate'] = trial.suggest_uniform(
                'other_rate',
                low=0.0,
                high=1.0
            )

            params['top_rate'] = trial.suggest_uniform(
                'top_rate',
                low=0.0,
                high=1.0-params['other_rate']
            )

        else:
            params['subsample'] = trial.suggest_uniform(
                'subsample',
                low=0.5,
                high=1.0
            )

            params['subsample_freq'] = trial.suggest_categorical(
                'subsample_freq',
                choices=[1]
            )

            if params['boosting_type'] == 'dart':
                params['drop_rate'] = trial.suggest_uniform(
                    'drop_rate',
                    low=0.0,
                    high=1.0
                )

                params['skip_drop'] = trial.suggest_uniform(
                    'skip_drop',
                    low=0.0,
                    high=1.0
                )

        return params

    def __call__(self, trial: optuna.trial.Trial) -> float:
        estimator = clone(self.estimator)
        params = self._get_params(trial)

        estimator.set_params(**params)

        scores = cross_val_score(
            estimator,
            self.X,
            self.y,
            cv=self.cv,
            error_score=self.error_score,
            groups=self.groups,
            scoring=self.scoring
        )

        return - np.average(scores)
