from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .compose import PipelineMaker
from .constants import MAIN_TABLE_NAME


class AutoMLClassifier(BaseEstimator, ClassifierMixin):
    _attributes = ['joiner_', 'sampler_', 'engineer_', 'search_cv_']
    _validate = False

    def __init__(
        self,
        info: Dict[str, Any],
        early_stopping_rounds: int = 10,
        lowercase: bool = False,
        max_iter: int = 10,
        n_estimators: int = 100,
        n_features_per_column: int = 32,
        n_jobs: int = -1,
        n_splits: int = 3,
        n_trials: int = 10,
        random_state: Union[int, np.random.RandomState] = 0,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        subsample: Union[int, float] = 1.0,
        valid_size: Union[int, float] = 0.25,
        verbose: int = 1
    ) -> None:
        super().__init__(verbose=verbose)

        self.early_stopping_rounds = early_stopping_rounds
        self.info = info
        self.lowercase = lowercase
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.n_features_per_column = n_features_per_column
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.subsample = subsample
        self.valid_size = valid_size

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        Xs: Dict[str, TWO_DIM_ARRAYLIKE_TYPE],
        y: ONE_DIM_ARRAYLIKE_TYPE,
        timeout: float = None
    ) -> 'AutoMLClassifier':
        related_tables = Xs.copy()
        X = related_tables.pop(MAIN_TABLE_NAME)

        index = X[self.info['time_col']].sort_values(na_position='first').index
        X = X.loc[index]
        y = y.loc[index]

        X, X_valid, y, y_valid = train_test_split(
            X,
            y,
            random_state=self.random_state,
            shuffle=False,
            test_size=self.valid_size
        )

        target_type = type_of_target(y)

        if target_type == 'binary':
            metric = 'auc'
            scoring = 'roc_auc'
        elif target_type in ['multiclass', 'multiclass-output']:
            metric = 'multiclass'
            scoring = 'f1_micro'
        else:
            raise ValueError(f'Unknown target_type: {target_type}.')

        maker = PipelineMaker(
            self.info,
            related_tables,
            target_type,
            cv=TimeSeriesSplit(self.n_splits),
            lowercase=self.lowercase,
            max_iter=self.max_iter,
            metric=metric,
            n_estimators=self.n_estimators,
            n_features_per_column=self.n_features_per_column,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            scoring=scoring,
            shuffle=False,
            subsample=self.subsample,
            timeout=None,
            verbose=self.verbose
        )

        self.joiner_ = maker.make_joiner()
        self.sampler_ = maker.make_sampler()
        self.engineer_ = maker.make_engineer()
        self.search_cv_ = maker.make_search_cv()

        X = self.joiner_.fit_transform(X)
        X_valid = self.joiner_.transform(X_valid)

        if self.sampler_ is not None:
            X, y = self.sampler_.fit_resample(X, y)

        X = self.engineer_.fit_transform(X)
        X_valid = self.engineer_.transform(X_valid)

        assert X.dtype == 'float32'

        model_name = self.search_cv_ \
            .estimator._final_estimator \
            .__class__.__name__.lower()
        fit_params = {
            f'{model_name}__early_stopping_rounds': self.early_stopping_rounds,
            f'{model_name}__eval_set': [(X_valid, y_valid)],
            f'{model_name}__verbose': False
        }

        self.search_cv_.fit(X, y, **fit_params)

        if target_type == 'binary':
            assert self.search_cv_.best_score_ > 0.5

        return self

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
