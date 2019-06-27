from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import numpy as np
import optuna

from imblearn.pipeline import make_pipeline
from joblib import Memory
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.impute import MissingIndicator
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import make_union

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .ensemble import LGBMClassifierCV
from .ensemble import LGBMRegressorCV
from .feature_extraction import TimeVectorizer
from .feature_selection import DropCollinearFeatures
from .feature_selection import FrequencyThreshold
from .feature_selection import NAProportionThreshold
from .impute import ModifiedSimpleImputer
from .preprocessing import ArithmeticalFeatures
from .preprocessing import Clip
from .preprocessing import CountEncoder
from .preprocessing import TextStatistics
from .table_join import get_categorical_feature_names
from .table_join import get_multi_value_categorical_feature_names
from .table_join import get_numerical_feature_names
from .table_join import get_time_feature_names
from .table_join import TableJoiner
from .under_sampling import ModifiedRandomUnderSampler


class BaseAutoMLModel(BaseEstimator):
    @property
    def best_iteration_(self) -> int:
        return self.model_.best_iteration_

    @property
    def best_params_(self) -> Dict[str, Any]:
        return self.model_.best_params_

    @property
    def best_score_(self) -> float:
        return self.model_.best_score_

    def __init__(
        self,
        cv: Union[int, BaseCrossValidator] = 5,
        dtype: Union[str, Type] = 'float32',
        info: Dict[str, Any] = None,
        learning_rate: float = 0.1,
        memory: Union[str, Memory] = None,
        n_estimators: int = 1_000,
        n_iter_no_change: int = 10,
        n_jobs: int = -1,
        n_seeds: int = 10,
        n_trials: int = 10,
        random_state: Union[int, np.random.RandomState] = 0,
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE] = None,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        study: optuna.study.Study = None,
        subsample: Union[int, float] = 1_000,
        timeout: float = None,
        verbose: int = 1
    ) -> None:
        super().__init__(verbose=verbose)

        self.cv = cv
        self.dtype = dtype
        self.info = info
        self.learning_rate = learning_rate
        self.memory = memory
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.random_state = random_state
        self.related_tables = related_tables
        self.sampling_strategy = sampling_strategy
        self.study = study
        self.subsample = subsample
        self.shuffle = shuffle
        self.timeout = timeout

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'BaseAutoMLModel':
        self.joiner_ = TableJoiner(
            info=self.info,
            related_tables=self.related_tables,
            verbose=self.verbose
        )
        self.engineer_ = self._make_mixed_transformer()
        self.sampler_ = self._make_sampler()
        self.model_ = self._make_model()

        X = self.joiner_.fit_transform(X)
        X = self.engineer_.fit_transform(X)

        if sample_weight is None:
            n_samples, _ = X.shape
            sample_weight = np.ones(n_samples)

        if self.sampler_ is not None:
            X, y = self.sampler_.fit_resample(X, y)
            sample_weight = sample_weight[self.sampler_.sample_indices_]

        self.model_.fit(X, y, sample_weight=sample_weight)

        return self

    def _make_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            FrequencyThreshold(verbose=self.verbose),
            make_union(
                make_pipeline(
                    ModifiedSimpleImputer(
                        n_jobs=self.n_jobs,
                        strategy='constant',
                        verbose=self.verbose
                    ),
                    CountEncoder(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    )
                ),
                MissingIndicator(error_on_new=False)
            )
        )

    def _make_multi_value_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            make_union(
                make_pipeline(
                    ModifiedSimpleImputer(
                        fill_value='',
                        n_jobs=self.n_jobs,
                        strategy='constant',
                        verbose=self.verbose
                    ),
                    make_union(
                        CountEncoder(
                            dtype=self.dtype,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                        ),
                        TextStatistics(
                            dtype=self.dtype,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                        )
                    )
                ),
                MissingIndicator(error_on_new=False)
            )
        )

    def _make_numerical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            FrequencyThreshold(
                max_frequency=np.iinfo('int32').max,
                verbose=self.verbose
            ),
            Clip(n_jobs=self.n_jobs, verbose=self.verbose),
            DropCollinearFeatures(
                subsample=self.subsample,
                verbose=self.verbose
            ),
            make_union(
                ArithmeticalFeatures(
                    dtype=self.dtype,
                    n_jobs=self.n_jobs,
                    operand=[],
                    verbose=self.verbose
                ),
                MissingIndicator(error_on_new=False)
            )
        )

    def _make_time_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            make_union(
                TimeVectorizer(
                    dtype=self.dtype,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                ),
                ArithmeticalFeatures(
                    dtype=self.dtype,
                    include_X=False,
                    n_jobs=self.n_jobs,
                    operand='subtract',
                    verbose=self.verbose
                )
            )
        )

    def _make_mixed_transformer(self) -> BaseEstimator:
        return make_column_transformer(
            (
                self._make_categorical_transformer(),
                get_categorical_feature_names
            ),
            (
                self._make_multi_value_categorical_transformer(),
                get_multi_value_categorical_feature_names
            ),
            (
                self._make_numerical_transformer(),
                get_numerical_feature_names
            ),
            (
                self._make_time_transformer(),
                get_time_feature_names
            )
        )

    def _make_sampler(self) -> BaseEstimator:
        if self._estimator_type == 'classifier':
            return ModifiedRandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                shuffle=self.shuffle,
                verbose=self.verbose
            )
        elif self._estimator_type == 'regressor':
            return None
        else:
            raise ValueError(
                f'Unknown _estimator_type: {self._estimator_type}.'
            )

    def _make_model(self) -> BaseEstimator:
        params = {
            'cv': self.cv,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'n_iter_no_change': self.n_iter_no_change,
            'n_jobs': self.n_jobs,
            'n_seeds': self.n_seeds,
            'n_trials': self.n_trials,
            'random_state': self.random_state,
            'study': self.study,
            'timeout': self.timeout,
            'verbose': self.verbose
        }

        if self._estimator_type == 'classifier':
            return LGBMClassifierCV(**params)
        elif self._estimator_type == 'regressor':
            return LGBMRegressorCV(**params)
        else:
            raise ValueError(
                f'Unknown _estimator_type: {self._estimator_type}.'
            )

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.model_.predict(X)


class AutoMLClassifier(BaseAutoMLModel, ClassifierMixin):
    """

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from automllib.automl import AutoMLClassifier
    >>> clf = AutoMLClassifier(n_iter_no_change=10, random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    AutoMLClassifier(...)
    >>> clf.score(X, y)
    0.9...
    """

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.model_.predict_proba(X)


class AutoMLRegressor(BaseAutoMLModel, RegressorMixin):
    """

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from automllib.automl import AutoMLRegressor
    >>> reg = AutoMLRegressor(n_iter_no_change=10, random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    AutoMLRegressor(...)
    >>> reg.score(X, y)
    0.9...
    """
