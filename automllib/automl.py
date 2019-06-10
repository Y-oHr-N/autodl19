from typing import Any
from typing import Dict
from typing import Union

import numpy as np

from joblib import Memory
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
        alpha: float = 0.005,
        cv: Union[int, BaseCrossValidator] = 5,
        early_stopping_rounds: int = 10,
        learning_rate: float = 0.01,
        lowercase: bool = False,
        max_depth: int = 7,
        max_iter: int = 10,
        memory: Union[str, Memory] = None,
        n_estimators: int = 100,
        n_features: int = 32,
        n_jobs: int = -1,
        n_trials: int = 10,
        random_state: Union[int, np.random.RandomState] = 0,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        subsample: Union[int, float] = 100_000,
        timeout: float = None,
        validation_fraction: Union[int, float] = 0.1,
        verbose: int = 1
    ) -> None:
        super().__init__(verbose=verbose)

        self.alpha = alpha
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.info = info
        self.learning_rate = learning_rate
        self.lowercase = lowercase
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.memory = memory
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
        self.validation_fraction = validation_fraction

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        timeout: float = None
    ) -> 'AutoMLModel':
        logger = self._get_logger()
        X = X.sort_values(self.info['time_col'], na_position='first')
        y = y.loc[X.index]
        fit_params = {}
        target_type = type_of_target(y)

        if timeout is None:
            timeout = self.timeout

        maker = KDDCup19Maker(
            self.info,
            self.related_tables,
            target_type,
            alpha=self.alpha,
            cv=self.cv,
            learning_rate=self.learning_rate,
            lowercase=self.lowercase,
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            memory=self.memory,
            n_estimators=self.n_estimators,
            n_features=self.n_features,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            shuffle=self.shuffle,
            subsample=self.subsample,
            timeout=timeout,
            verbose=self.verbose
        )

        self.joiner_ = maker.make_joiner()
        self.engineer_ = maker.make_mixed_transformer()
        self.drift_dropper_ = maker.make_sampler()
        self.sampler_ = maker.make_sampler()
        self.search_cv_ = maker.make_search_cv()

        X = self.joiner_.fit_transform(X)

        if self.validation_fraction > 0.0:
            X, X_valid, y, y_valid = train_test_split(
                X,
                y,
                random_state=self.random_state,
                shuffle=self.shuffle,
                test_size=self.validation_fraction
            )

        X = self.engineer_.fit_transform(X)

        assert X.dtype == 'float32'

        if self.validation_fraction > 0.0:
            X_valid = self.engineer_.transform(X_valid)

            # X = self.drift_dropper_.fit_transform(X, X_test=X_valid)
            # X_valid = self.drift_dropper_.transform(X_valid)

            fit_params['early_stopping_rounds'] = self.early_stopping_rounds
            fit_params['eval_set'] = [(X_valid, y_valid)]
            fit_params['verbose'] = False

        if self.sampler_ is not None:
            X, y = self.sampler_.fit_resample(X, y)

        self.search_cv_.fit(X, y, **fit_params)

        logger.info(f'The CV score is {self.best_score_:.3f}.')

        if self.validation_fraction > 0.0:
            logger.info(
                f'The validation score is '
                f'{self.search_cv_.score(X_valid, y_valid):.3f}.'
            )

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'no_validation': True}

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        # if self.validation_fraction > 0.0:
        #     X = self.drift_dropper_.transform(X)

        return self.search_cv_.predict(X)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        # if self.validation_fraction > 0.0:
        #     X = self.drift_dropper_.transform(X)

        return self.search_cv_.predict_proba(X)

    def score(self, X, y):
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        # if self.validation_fraction > 0.0:
        #     X = self.drift_dropper_.transform(X)

        return self.search_cv_.score(X)
