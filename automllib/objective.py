from typing import Any
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score


class Objective(object):
    def __init__(
        self,
        estimator: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> None:
        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv

    def _get_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        params = {
            'base_estimator__boosting_type': trial.suggest_categorical(
                'base_estimator__boosting_type',
                choices=['gbdt', 'goss', 'dart', 'rf']
            ),
            'base_estimator__num_leaves': trial.suggest_int(
                'base_estimator__num_leaves',
                low=2,
                high=123
            ),
            'base_estimator__reg_alpha': trial.suggest_loguniform(
                'base_estimator__reg_alpha',
                low=1e-06,
                high=10.0
            ),
            'base_estimator__reg_lambda': trial.suggest_loguniform(
                'base_estimator__reg_lambda',
                low=1e-06,
                high=10.0
            ),
            'base_estimator__subsample': trial.suggest_uniform(
                'base_estimator__subsample',
                low=0.5,
                high=1.0
            )
        }

        if params['base_estimator__boosting_type'] == 'goss':
            params['base_estimator__top_rate'] = trial.suggest_uniform(
                'base_estimator__top_rate',
                low=0.0,
                high=1.0
            )
            params['base_estimator__other_rate'] = trial.suggest_uniform(
                'base_estimator__other_rate',
                low=0.0,
                high=1.0-params['base_estimator__top_rate']
            )
        else:
            params['base_estimator__colsample_bytree'] = trial.suggest_uniform(
                'base_estimator__colsample_bytree',
                low=0.5,
                high=1.0
            )

            if params['base_estimator__boosting_type'] == 'dart':
                params['base_estimator__drop_rate'] = trial.suggest_uniform(
                    'base_estimator__drop_rate',
                    low=0.0,
                    high=1.0
                )
                params['base_estimator__skip_drop'] = trial.suggest_uniform(
                    'base_estimator__skip_drop',
                    low=0.0,
                    high=1.0
                )

        return params

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = self._get_params(trial)

        self.estimator.set_params(**params)

        scores = cross_val_score(
            self.estimator,
            self.X,
            self.y,
            cv=self.cv,
            error_score='raise',
            scoring='roc_auc'
        )

        return - np.average(scores)
