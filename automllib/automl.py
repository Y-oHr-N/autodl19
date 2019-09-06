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
from .feature_selection import FeatureSelector
from .impute import ModifiedSimpleImputer
from .preprocessing import ArithmeticalFeatures
from .preprocessing import ArithmeticalFeatures_1
from .preprocessing import Clip
from .preprocessing import CountEncoder
from .preprocessing import TextStatistics
from .table_join import get_categorical_feature_names
from .table_join import get_multi_value_categorical_feature_names
from .table_join import get_numerical_feature_names
from .table_join import get_time_feature_names
from .table_join import TableJoiner
from .under_sampling import RandomUniformSampler


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
        operand: Union[Sequence[str], str] = ['add', 'subtract', 'multiply'],
        subsample: Union[float, int] = 1_000,
        threshold: float = 0.6,
        max_n_combinations: int = 300,
        # Parameters for a selector
        train_size: float = 0.8,  # Use 80% data for training
        train_size_for_searching: float = 0.4,  # Use 40% train data for tuning
        valid_size: float = 0.2,  # Use 20% tuning data for validation
        select_study: optuna.study.Study = None,
        importance_type: str = 'gain',
        gain_threshold: float = 0.0,
        # Parameters for a sampler
        sampling_strategy: Union[Dict[str, int], float, str] = 'auto',
        max_samples: int = 100_000,
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
        self.max_samples = max_samples
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
        self.train_size = train_size
        self.train_size_for_searching = train_size_for_searching
        self.valid_size = valid_size
        self.select_study = select_study
        self.gain_threshold = gain_threshold
        self.max_n_combinations = max_n_combinations
        self.importance_type = importance_type

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE] = None
    ) -> 'BaseAutoMLModel':
        if self.time_col is not None:
            indices = np.argsort(X[self.time_col].values)
            X = safe_indexing(X, indices)
            y = safe_indexing(y, indices)

        self.joiner_ = TableJoiner(
            relations=self.relations,
            tables=self.tables,
            time_col=self.time_col,
            verbose=self.verbose
        )

        a = []

        self.engineer_ = self._make_mixed_transformer()
        self.selector_ = self._make_selector(a)
        self.sampler_ = self._make_sampler()
        self.engineer_ = self._make_mixed_transformer()
        self.model_ = self._make_model()
        self.second_engineer_ = self._make_second_engineer_(a)

        X = self.joiner_.fit_transform(X, related_tables=related_tables)
        X, y = self.sampler_.fit_resample(X, y)
        X = self.engineer_.fit_transform(X)
        X = self.selector_.fit_transform(X, y)
        X = self.second_engineer_.fit_transform(X)

        self.model_.fit(X, y)

        return self

    def _make_joiner(self) -> BaseEstimator:
        return TableJoiner(
            relations=self.relations,
            tables=self.tables,
            time_col=self.time_col,
            verbose=self.verbose
        )

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
                # ArithmeticalFeatures(
                #     dtype=self._dtype,
                #     n_jobs=self.n_jobs,
                #     operand='subtract',
                #     verbose=self.verbose
                # ),
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

    def _make_selector(self,a) -> BaseEstimator:
        return FeatureSelector(
                time_col=self.time_col,
                train_size=self.train_size,
                train_size_for_searching=self.train_size_for_searching,
                valid_size=self.valid_size,
                learning_rate=self.learning_rate,
                num_boost_round=self.n_iter_no_change,
                early_stopping_rounds=self.n_estimators,
                n_trials=self.n_trials,
                study=self.select_study,
                importance_type=self.importance_type,
                random_state=self.random_state,
                gain_threshold=self.gain_threshold,
                verbose=self.verbose,
                a=a
            )

    def _make_second_engineer_(self,a) -> BaseEstimator:
        return ArithmeticalFeatures_1(
            dtype=self._dtype,
            include_X=False,
            n_jobs=self.n_jobs,
            operand=self.operand,
            max_n_combinations=self.max_n_combinations,
            verbose=self.verbose,
            a=a
        )

    def _make_sampler(self) -> BaseEstimator:
        return RandomUniformSampler(
            random_state=self.random_state,
            shuffle=False,
            subsample=self.max_samples,
            time_col=self.time_col,
            verbose=self.verbose
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
        X = self.selector_.transform(X)
        X = self.second_engineer_.transform(X)

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
        X = self.selector_.transform(X)
        X = self.second_engineer_.transform(X)

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
