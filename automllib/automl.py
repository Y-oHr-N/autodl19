from typing import Any
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
import optuna

from imblearn.pipeline import make_pipeline
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.impute import MissingIndicator
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_union
from sklearn.utils import safe_indexing

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
    _dtype = 'float32'

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
        n_jobs: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        verbose: int = 0,
        # Parameters for a joiner
        relations: Sequence[Dict[str, str]] = None,
        tables: Dict[str, Dict[str, str]] = None,
        time_col: str = None,
        # Parameters for an engineer
        operand: Union[Sequence[str], str] = None,
        subsample: Union[float, int] = 1_000,
        threshold: float = 0.6,
        # Parameters for a sampler
        sampling_strategy: Union[Dict[str, int], float, str] = 'auto',
        # Parameters for a model
        class_weight: Union[str, Dict[str, float]] = 'balanced',
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = True,
        learning_rate: float = 0.1,
        n_estimators: int = 1_000,
        n_iter_no_change: int = 10,
        n_trials: int = 10,
        n_seeds: int = 10,
        objective: str = None,
        study: optuna.study.Study = None,
        timeout: float = None,
        **kwargs: Any
    ) -> None:
        super().__init__(verbose=verbose)

        self.class_weight = class_weight
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.objective = objective
        self.operand = operand
        self.random_state = random_state
        self.relations = relations
        self.sampling_strategy = sampling_strategy
        self.study = study
        self.subsample = subsample
        self.tables = tables
        self.threshold = threshold
        self.timeout = timeout
        self.time_col = time_col

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None,
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE] = None
    ) -> 'BaseAutoMLModel':
        if sample_weight is None:
            n_samples, _ = X.shape
            sample_weight = np.ones(n_samples)

        if self.time_col is not None:
            indices = np.argsort(X[self.time_col].values)
            X = safe_indexing(X, indices)
            y = safe_indexing(y, indices)
            sample_weight = safe_indexing(sample_weight, indices)

        self.joiner_ = TableJoiner(
            relations=self.relations,
            tables=self.tables,
            time_col=self.time_col,
            verbose=self.verbose
        )
        self.engineer_ = self._make_mixed_transformer()
        self.sampler_ = self._make_sampler()
        self.model_ = self._make_model()

        X = self.joiner_.fit_transform(X, related_tables=related_tables)
        X = self.engineer_.fit_transform(X)

        # if self.sampler_ is not None:
        #     X, y = self.sampler_.fit_resample(X, y)
        #     sample_weight = safe_indexing(
        #         sample_weight,
        #         self.sampler_.sample_indices_
        #     )

        self.model_.fit(X, y, sample_weight=sample_weight)

        return self

    def _make_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(
                threshold=self.threshold,
                verbose=self.verbose
            ),
            FrequencyThreshold(verbose=self.verbose),
            make_union(
                make_pipeline(
                    ModifiedSimpleImputer(
                        n_jobs=self.n_jobs,
                        strategy='constant',
                        verbose=self.verbose
                    ),
                    CountEncoder(
                        dtype=self._dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    )
                ),
                MissingIndicator(error_on_new=False)
            )
        )

    def _make_multi_value_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(
                threshold=self.threshold,
                verbose=self.verbose
            ),
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
                            dtype=self._dtype,
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                        ),
                        TextStatistics(
                            dtype=self._dtype,
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
            NAProportionThreshold(
                threshold=self.threshold,
                verbose=self.verbose
            ),
            FrequencyThreshold(
                max_frequency=None,
                verbose=self.verbose
            ),
            Clip(n_jobs=self.n_jobs, verbose=self.verbose),
            DropCollinearFeatures(
                random_state=self.random_state,
                subsample=self.subsample,
                verbose=self.verbose
            ),
            make_union(
                ArithmeticalFeatures(
                    dtype=self._dtype,
                    n_jobs=self.n_jobs,
                    operand=self.operand,
                    verbose=self.verbose
                ),
                MissingIndicator(error_on_new=False)
            )
        )

    def _make_time_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(
                threshold=self.threshold,
                verbose=self.verbose
            ),
            make_union(
                TimeVectorizer(
                    dtype=self._dtype,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                ),
                ArithmeticalFeatures(
                    dtype=self._dtype,
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
                shuffle=False,
                verbose=self.verbose
            )
        elif self._estimator_type == 'regressor':
            return None
        else:
            raise ValueError(
                f'Unknown _estimator_type: {self._estimator_type}.'
            )

    def _make_model(self) -> BaseEstimator:
        if isinstance(self.cv, int) and self.time_col is not None:
            cv = TimeSeriesSplit(self.cv)
        else:
            cv = self.cv

        params = {
            'class_weight': self.class_weight,
            'cv': cv,
            'enable_pruning': self.enable_pruning,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'n_iter_no_change': self.n_iter_no_change,
            'n_jobs': self.n_jobs,
            'n_seeds': self.n_seeds,
            'n_trials': self.n_trials,
            'objective': self.objective,
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
        """Predict using the Fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """

        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.model_.predict(X)


class AutoMLClassifier(BaseAutoMLModel, ClassifierMixin):
    """AutoML classifier.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from automllib.automl import AutoMLClassifier
    >>> clf = AutoMLClassifier(random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    AutoMLClassifier(...)
    >>> clf.score(X, y)
    0.9...
    """

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        """Predict class probabilities for data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        p
            Class probabilities of data.
        """

        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)

        return self.model_.predict_proba(X)


class AutoMLRegressor(BaseAutoMLModel, RegressorMixin):
    """AutoML regressor.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from automllib.automl import AutoMLRegressor
    >>> reg = AutoMLRegressor(random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    AutoMLRegressor(...)
    >>> reg.score(X, y)
    0.9...
    """
