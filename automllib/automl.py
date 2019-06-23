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
from sklearn.utils.multiclass import type_of_target

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .ensemble import LGBMClassifierCV
from .ensemble import LGBMRegressorCV
# from .feature_extraction import MultiValueCategoricalVectorizer
from .feature_extraction import TimeVectorizer
from .feature_selection import DropCollinearFeatures
from .feature_selection import FrequencyThreshold
from .feature_selection import NAProportionThreshold
from .impute import ModifiedSimpleImputer
from .model_selection import OptunaSearchCV
from .preprocessing import ArithmeticalFeatures
from .preprocessing import Clip
from .preprocessing import CountEncoder
# from .preprocessing import RowStatistics
# from .preprocessing import ModifiedStandardScaler
from .preprocessing import TextStatistics
from .table_join import get_categorical_feature_names
from .table_join import get_multi_value_categorical_feature_names
from .table_join import get_numerical_feature_names
from .table_join import get_time_feature_names
from .table_join import TableJoiner
from .under_sampling import ModifiedRandomUnderSampler


class AutoMLModel(BaseEstimator):
    @property
    def best_params_(self) -> Dict[str, Any]:
        return self.model_._final_estimator.best_params_

    @property
    def best_score_(self) -> float:
        return self.model_._final_estimator.best_score_

    def __init__(
        self,
        info: Dict[str, Any],
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE],
        cv: Union[int, BaseCrossValidator] = 5,
        dtype: Union[str, Type] = 'float32',
        learning_rate: float = 0.1,
        lowercase: bool = False,
        max_iter: int = 10,
        memory: Union[str, Memory] = None,
        n_estimators: int = 300,
        n_features: int = 32,
        n_iter_no_change: int = 10,
        n_jobs: int = -1,
        n_trials: int = 100,
        random_state: Union[int, np.random.RandomState] = 0,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        subsample: Union[int, float] = 1.0,
        timeout: float = None,
        verbose: int = 1
    ) -> None:
        super().__init__(verbose=verbose)

        self.cv = cv
        self.dtype = dtype
        self.info = info
        self.learning_rate = learning_rate
        self.lowercase = lowercase
        self.max_iter = max_iter
        self.memory = memory
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.related_tables = related_tables
        self.sampling_strategy = sampling_strategy
        self.shuffle = shuffle
        self.subsample = subsample
        self.timeout = timeout

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        timeout: float = None
    ) -> 'AutoMLModel':
        X = X.sort_values(self.info['time_col'], na_position='first')
        y = y.loc[X.index]

        self.target_type = type_of_target(y)
        self.model_ = make_pipeline(
            self.make_joiner(),
            self.make_mixed_transformer(),
            self.make_sampler(),
            self.make_model()
        )

        self.model_.fit(X, y)

        return self

    def make_joiner(self) -> BaseEstimator:
        return TableJoiner(
            self.info,
            self.related_tables,
            verbose=self.verbose
        )

    def make_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
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
            NAProportionThreshold(verbose=self.verbose),
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
            NAProportionThreshold(verbose=self.verbose),
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
            #         ArithmeticalFeatures(
            #             n_jobs=self.n_jobs,
            #             operand=['subtract', 'polynomial'],
            #             verbose=self.verbose
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
            NAProportionThreshold(verbose=self.verbose),
            make_union(
                make_pipeline(
                    TimeVectorizer(
                        dtype=self.dtype,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    FrequencyThreshold(verbose=self.verbose),
                    DropCollinearFeatures(verbose=self.verbose)
                ),
                make_pipeline(
            #         ModifiedSimpleImputer(
            #             n_jobs=self.n_jobs,
            #             strategy='min',
            #             verbose=self.verbose
            #         ),
                    ArithmeticalFeatures(
                        dtype=self.dtype,
                        include_X=False,
                        n_jobs=self.n_jobs,
                        operand='subtract',
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

    def make_sampler(self) -> BaseEstimator:
        if self.target_type in ['binary', 'multiclass', 'multiclass-output']:
            return ModifiedRandomUnderSampler(
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
            'cv': self.cv,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'n_iter_no_change': self.n_iter_no_change,
            'n_jobs': self.n_jobs,
            'n_trials': self.n_trials,
            'random_state': self.random_state,
            'timeout': self.timeout,
            'verbose': self.verbose
        }

        if self.target_type in ['binary', 'multiclass', 'multiclass-output']:
            return LGBMClassifierCV(**params)
        elif self.target_type in ['continuous', 'continuous-output']:
            return LGBMRegressorCV(**params)
        else:
            raise ValueError(f'Unknown target_type: {self.target_type}.')

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        return self.model_.predict(X)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        return self.model_.predict_proba(X)

    def score(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE
    ) -> float:
        self._check_is_fitted()

        return self.model_.score(X)
