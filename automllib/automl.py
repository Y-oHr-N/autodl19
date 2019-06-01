from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from joblib import Memory
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .compose import KDDCup19Maker


class AutoMLModel(BaseEstimator):
    _attributes = ['engineer_', 'joiner_', 'sampler_', 'search_cv_']

    @property
    def best_params_(self) -> Dict[str, Any]:
        return self.search_cv_.best_params_

    @property
    def best_score_(self) -> float:
        return self.search_cv_.best_score_

    def __init__(
        self,
        info: Dict[str, Any],
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE],
        cv: Union[int, BaseCrossValidator] = 5,
        early_stopping_rounds: int = 10,
        learning_rate: float = 0.01,
        lowercase: bool = True,
        max_depth: int = 5,
        max_features: int = 100,
        max_iter: int = 10,
        memory: Union[str, Memory] = None,
        n_estimators: int = 100,
        n_jobs: int = 1,
        n_trials: int = 10,
        random_state: Union[int, np.random.RandomState] = None,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        subsample: Union[int, float] = 1.0,
        validation_fraction: Union[int, float] = 0.25,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.info = info
        self.learning_rate = learning_rate
        self.lowercase = lowercase
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.memory = memory
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.related_tables = related_tables
        self.sampling_strategy = sampling_strategy
        self.shuffle = shuffle
        self.subsample = subsample
        self.validation_fraction = validation_fraction

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        timeout: float = None
    ) -> 'AutoMLModel':
        target_type = type_of_target(y)
        maker = KDDCup19Maker(
            self.info,
            self.related_tables,
            target_type,
            cv=self.cv,
            learning_rate=self.learning_rate,
            lowercase=self.lowercase,
            max_depth=self.max_depth,
            max_features=self.max_features,
            max_iter=self.max_iter,
            memory=self.memory,
            n_estimators=self.n_estimators,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            shuffle=self.shuffle,
            subsample=self.subsample,
            # timeout=None,
            verbose=self.verbose
        )

        self.joiner_ = maker.make_joiner()
        self.sampler_ = maker.make_sampler()
        self.engineer_ = maker.make_mixed_transformer()
        self.search_cv_ = maker.make_search_cv()

        X = X.sort_values(self.info['time_col'], na_position='first')
        y = y.loc[X.index]
        fit_params = {}

        X = self.joiner_.fit_transform(X)

        if self.validation_fraction > 0.0:
            X, X_valid, y, y_valid = train_test_split(
                X,
                y,
                random_state=self.random_state,
                shuffle=self.shuffle,
                test_size=self.validation_fraction
            )

        if self.sampler_ is not None:
            X, y = self.sampler_.fit_resample(X, y)

        X = self.engineer_.fit_transform(X)

        assert X.dtype == 'float32'

        if self.validation_fraction > 0.0:
            X_valid = self.engineer_.transform(X_valid)

            model_name = self.search_cv_ \
                .estimator._final_estimator \
                .__class__.__name__.lower()

            fit_params[f'{model_name}__early_stopping_rounds'] = \
                self.early_stopping_rounds
            fit_params[f'{model_name}__eval_set'] = [(X_valid, y_valid)]
            fit_params[f'{model_name}__verbose'] = False

        self.search_cv_.fit(X, y, **fit_params)

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'no_validation': True}

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.search_cv_.predict(X)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.search_cv_.predict_proba(X)

    def score(self, X, y):
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.search_cv_.score(X)
