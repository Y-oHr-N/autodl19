from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Union

import numpy as np
import optuna

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.model_selection import cross_val_score

from .utils import timeit


class Objective(object):
    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Mapping[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv: Union[str, int, BaseCrossValidator, None] = 5,
        error_score: Union[str, float] = np.nan,
        fit_params: Dict[str, Any] = None,
        groups: np.ndarray = None,
        scoring: Union[str, Callable] = None
    ) -> None:
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.X = X
        self.y = y
        self.cv = cv
        self.error_score = error_score
        self.fit_params = fit_params
        self.groups = groups
        self.scoring = scoring

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
            fit_params=self.fit_params,
            groups=self.groups,
            scoring=self.scoring
        )

        return - np.average(scores)

    def _get_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        return {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_distributions.items()
        }


class OptunaSearchCV(BaseEstimator):
    @property
    def best_index_(self) -> int:
        df = self.trials_dataframe()

        return df['value'].idxmin()

    @property
    def best_params_(self) -> Dict[str, Any]:
        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        return - self.best_value_

    @property
    def best_trial_(self) -> optuna.structs.FrozenTrial:
        return self.study_.best_trial

    @property
    def best_value_(self) -> float:
        return self.study_.best_value

    @property
    def predict(self) -> Callable:
        return self.best_estimator_.predict

    @property
    def predict_proba(self) -> Callable:
        return self.best_estimator_.predict_proba

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Mapping,
        cv: Union[str, int, BaseCrossValidator, None] = 5,
        error_score: Union[str, float] = np.nan,
        load_if_exists: bool = False,
        n_jobs: int = 1,
        n_trials: int = 10,
        refit: bool = True,
        sampler: optuna.samplers.BaseSampler = None,
        scoring: Union[str, Callable] = None,
        storage: Union[str, optuna.storages.BaseStorage] = None,
        study_name: str = None,
        timeout: float = None
    ) -> None:
        self.cv = cv
        self.error_score = error_score
        self.estimator = estimator
        self.load_if_exists = load_if_exists
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions
        self.refit = refit
        self.sampler = sampler
        self.scoring = scoring
        self.storage = storage
        self.study_name = study_name
        self.timeout = timeout

    def _refit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        **fit_params: Dict[str, Any]
    ) -> 'OptunaSearchCV':
        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**self.study_.best_params)

        self.best_estimator_.fit(X, y, **fit_params)

        return self

    @timeit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None,
        **fit_params: Dict[str, Any]
    ) -> 'OptunaSearchCV':
        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)

        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self.study_ = optuna.create_study(
            load_if_exists=self.load_if_exists,
            sampler=self.sampler,
            storage=self.storage,
            study_name=self.study_name
        )

        objective = Objective(
            self.estimator,
            self.param_distributions,
            X,
            y,
            cv=cv,
            error_score=self.error_score,
            fit_params=fit_params,
            groups=groups,
            scoring=self.scorer_
        )

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        if self.refit:
            self._refit(X, y, **fit_params)

        return self
