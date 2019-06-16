from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna

from imblearn.pipeline import make_pipeline
from joblib import Memory
from sklearn.compose import make_column_transformer
# from sklearn.experimental import enable_iterative_imputer  # noqa
# from sklearn.impute import IterativeImputer
# from sklearn.impute import MissingIndicator
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_union
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.multiclass import type_of_target

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE
# from .feature_extraction import MultiValueCategoricalVectorizer
from .feature_extraction import TimeVectorizer
from .feature_selection import DropCollinearFeatures
from .feature_selection import DropDriftFeatures
from .feature_selection import FrequencyThreshold
# from .feature_selection import NAProportionThreshold
from .impute import ModifiedSimpleImputer
from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .preprocessing import CountEncoder
# from .preprocessing import RowStatistics
# from .preprocessing import ModifiedStandardScaler
from .preprocessing import SubtractedFeatures
from .preprocessing import TextStatistics
from .table_join import get_categorical_feature_names
from .table_join import get_multi_value_categorical_feature_names
from .table_join import get_numerical_feature_names
from .table_join import get_time_feature_names
from .table_join import TableJoiner
from .under_sampling import RandomUnderSampler


class AutoMLModel(BaseEstimator):
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
        dtype: Union[str, Type] = 'float32',
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
        subsample: Union[int, float] = 1.0,
        timeout: float = None,
        validation_fraction: Union[int, float] = 0.1,
        verbose: int = 1
    ) -> None:
        super().__init__(verbose=verbose)

        self.alpha = alpha
        self.cv = cv
        self.dtype = dtype
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
        X_valid = None
        y_valid = None
        fit_params = {}
        self.target_type = type_of_target(y)

        if timeout is None:
            timeout = self.timeout

        self.joiner_ = self.make_joiner()
        self.engineer_ = self.make_mixed_transformer()
        self.drift_dropper_ = self.make_selector()
        self.sampler_ = self.make_sampler()
        self.search_cv_ = self.make_search_cv()

        X = self.joiner_.fit_transform(X)
        X = self.engineer_.fit_transform(X)

        assert X.dtype == 'float32'

        if self.validation_fraction > 0.0:
            X, X_valid, y, y_valid = train_test_split(
                X,
                y,
                random_state=self.random_state,
                shuffle=self.shuffle,
                test_size=self.validation_fraction
            )

        X = self.drift_dropper_.fit_transform(X, X_test=X_valid)

        if self.validation_fraction > 0.0:
            X_valid = self.drift_dropper_.transform(X_valid)
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

    def make_joiner(self) -> BaseEstimator:
        return TableJoiner(
            self.info,
            self.related_tables,
            verbose=self.verbose
        )

    def make_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            # NAProportionThreshold(verbose=self.verbose),
            FrequencyThreshold(verbose=self.verbose),
            CountEncoder(
                dtype=self.dtype,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            ),
            DropCollinearFeatures(verbose=self.verbose),
            memory=self.memory
        )

    def make_multi_value_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            # NAProportionThreshold(verbose=self.verbose),
            ModifiedSimpleImputer(
                fill_value='',
                n_jobs=self.n_jobs,
                strategy='constant',
                verbose=self.verbose
            ),
            make_union(
            #     MultiValueCategoricalVectorizer(
            #         dtype=self.dtype,
            #         lowercase=self.lowercase,
            #         n_features=self.n_features,
            #         n_jobs=self.n_jobs,
            #         verbose=self.verbose
            #     ),
                make_pipeline(
                    CountEncoder(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    DropCollinearFeatures(verbose=self.verbose)
                ),
                make_pipeline(
                    TextStatistics(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    DropCollinearFeatures(verbose=self.verbose)
                )
            ),
            memory=self.memory
        )

    def make_numerical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            # NAProportionThreshold(verbose=self.verbose),
            FrequencyThreshold(
                max_frequency=np.iinfo('int64').max,
                verbose=self.verbose
            ),
            # make_union(
                make_pipeline(
                    Clip(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    DropCollinearFeatures(verbose=self.verbose),
            #         ModifiedStandardScaler(
            #             n_jobs=self.n_jobs,
            #             verbose=self.verbose
            #         ),
            #         IterativeImputer(
            #             estimator=LinearRegression(n_jobs=self.n_jobs),
            #             max_iter=self.max_iter
            #         ),
            #         make_union(
            #             PolynomialFeatures(
            #                 include_bias=False,
            #                 interaction_only=True
            #             ),
            #             SubtractedFeatures(
            #                 n_jobs=self.n_jobs,
            #                 verbose=self.verbose
            #             )
            #         )
                ),
            #     make_pipeline(
            #         CountEncoder(
            #             dtype=self.dtype,
            #             n_jobs=self.n_jobs,
            #             verbose=self.verbose
            #         ),
            #         DropCollinearFeatures(verbose=self.verbose)
            #     )
            #     MissingIndicator(error_on_new=False),
            #     RowStatistics(
            #         dtype=self.dtype,
            #         n_jobs=self.n_jobs,
            #         verbose=self.verbose
            #     )
            # ),
            memory=self.memory
        )

    def make_time_transformer(self) -> BaseEstimator:
        return make_pipeline(
            # NAProportionThreshold(verbose=self.verbose),
            make_union(
                make_pipeline(
                    TimeVectorizer(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    DropCollinearFeatures(verbose=self.verbose)
                ),
                make_pipeline(
            #         ModifiedSimpleImputer(
            #             n_jobs=self.n_jobs,
            #             strategy='min',
            #             verbose=self.verbose
            #         ),
                    SubtractedFeatures(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
            #         DropCollinearFeatures(verbose=self.verbose)
                )
            ),
            memory=self.memory
        )

    def make_mixed_transformer(self) -> BaseEstimator:
        return make_column_transformer(
            (
                self.make_categorical_transformer(),
                get_categorical_feature_names
            ),
            (
                self.make_multi_value_categorical_transformer(),
                get_multi_value_categorical_feature_names
            ),
            (
                self.make_numerical_transformer(),
                get_numerical_feature_names
            ),
            (
                self.make_time_transformer(),
                get_time_feature_names
            )
        )

    def make_selector(self) -> BaseEstimator:
        return DropDriftFeatures(
            alpha=self.alpha,
            random_state=self.random_state,
            verbose=self.verbose
        )

    def make_sampler(self) -> BaseEstimator:
        if self.target_type in ['binary', 'multiclass', 'multiclass-output']:
            return RandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                shuffle=self.shuffle,
                verbose=self.verbose
            )
        elif self.target_type in ['continuous', 'continuous-output']:
            return None
        else:
            raise ValueError(f'Unknown target_type: {self.target_type}.')

    def make_model(self) -> BaseEstimator:
        params = {
            'learning_rate': self.learning_rate,
            # 'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'n_jobs': 1,
            'random_state': self.random_state,
            'subsample_freq': 1
        }

        if self.target_type in ['binary', 'multiclass', 'multiclass-output']:
            if self.target_type == 'binary':
                params['is_unbalance'] = True
                params['metric'] = 'auc'
            else:
                params['class_weight'] = 'balanced'

            return lgb.LGBMClassifier(**params)

        elif self.target_type in ['continuous', 'continuous-output']:
            return lgb.LGBMRegressor(**params)

        else:
            raise ValueError(f'Unknown target_type: {self.target_type}.')

    def make_search_cv(self, timeout: float = None) -> BaseEstimator:
        model = self.make_model()
        param_distributions = {
            'colsample_bytree':
                optuna.distributions.DiscreteUniformDistribution(
                    0.1,
                    1.0,
                    0.1
                ),
            'min_child_samples':
                optuna.distributions.IntUniformDistribution(1, 100),
            'min_child_weight':
                optuna.distributions.LogUniformDistribution(1e-03, 10.0),
            'num_leaves':
                optuna.distributions.IntUniformDistribution(
                    2,
                    2 ** self.max_depth - 1
                ),
            'reg_alpha':
                optuna.distributions.LogUniformDistribution(1e-06, 10.0),
            'reg_lambda':
                optuna.distributions.LogUniformDistribution(1e-06, 10.0),
            'subsample':
                optuna.distributions.DiscreteUniformDistribution(0.1, 1.0, 0.1)
        }

        if self.target_type == 'binary':
            scoring = 'roc_auc'
        else:
            scoring = None

        if timeout is None:
            timeout = self.timeout

        return OptunaSearchCV(
            model,
            param_distributions,
            cv=self.cv,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            scoring=scoring,
            subsample=self.subsample,
            timeout=timeout,
            verbose=self.verbose
        )

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)
        X = self.drift_dropper_.transform(X)

        return self.search_cv_.predict(X)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)
        X = self.drift_dropper_.transform(X)

        return self.search_cv_.predict_proba(X)

    def score(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE
    ) -> float:
        self._check_is_fitted()

        X = self.joiner_.transform(X)
        X = self.engineer_.transform(X)
        X = self.drift_dropper_.transform(X)

        return self.search_cv_.score(X)
