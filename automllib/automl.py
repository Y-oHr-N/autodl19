from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .compose import PipelineMaker


class AutoMLModel(BaseEstimator):
    _attributes = ['engineer_', 'joiner_', 'sampler_', 'search_cv_']
    _validate = False

    def __init__(
        self,
        info: Dict[str, Any],
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE],
        cv: Union[int, BaseCrossValidator] = 5,
        early_stopping_rounds: int = 10,
        lowercase: bool = False,
        max_depth: int = 7,
        max_iter: int = 10,
        n_estimators: int = 100,
        n_features: int = 1_048_576,
        n_jobs: int = 1,
        n_trials: int = 10,
        random_state: Union[int, np.random.RandomState] = None,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        subsample: Union[int, float] = 1.0,
        timeout: float = None,
        valid_size: Union[int, float] = 0.25,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.info = info
        self.lowercase = lowercase
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.related_tables = related_tables
        self.sampling_strategy = sampling_strategy
        self.shuffle = shuffle
        self.subsample = subsample
        self.timeout = timeout
        self.valid_size = valid_size

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE
    ) -> 'AutoMLModel':
        target_type = type_of_target(y)

        if target_type == 'binary':
            metric = 'auc'
            scoring = 'roc_auc'
        else:
            metric = ''
            scoring = None

        maker = PipelineMaker(
            self.info,
            self.related_tables,
            target_type,
            cv=self.cv,
            lowercase=self.lowercase,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            metric=metric,
            n_estimators=self.n_estimators,
            n_features=self.n_features,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            scoring=scoring,
            shuffle=self.shuffle,
            subsample=self.subsample,
            timeout=None,
            verbose=self.verbose
        )

        self.joiner_ = maker.make_joiner()
        self.sampler_ = maker.make_sampler()
        self.engineer_ = maker.make_engineer()
        self.search_cv_ = maker.make_search_cv()

        if not self.shuffle:
            X = X.sort_values(self.info['time_col'])
            y = y.loc[X.index]

        X = self.joiner_.fit_transform(X)

        X, X_valid, y, y_valid = train_test_split(
            X,
            y,
            random_state=self.random_state,
            shuffle=self.shuffle,
            test_size=self.valid_size
        )

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


class AutoMLClassifier(AutoMLModel, ClassifierMixin):
    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.search_cv_.predict_proba(X)


class AutoMLRegressor(AutoMLModel, RegressorMixin):
    pass
