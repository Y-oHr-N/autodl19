from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna

from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFpr
from sklearn.impute import IterativeImputer
from sklearn.impute import MissingIndicator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import make_union
from sklearn.preprocessing import PolynomialFeatures

from .base import BaseEstimator
from .base import TWO_DIM_ARRAYLIKE_TYPE
from .feature_extraction import MultiValueCategoricalVectorizer
# from .feature_selection import DropDuplicates
from .feature_selection import DropCollinearFeatures
from .feature_selection import DropInvariant
from .feature_selection import DropUniqueKey
from .feature_selection import NAProportionThreshold
from .impute import SimpleImputer
from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .preprocessing import CountEncoder
from .preprocessing import RowStatistics
from .preprocessing import StandardScaler
from .preprocessing import SubtractedFeatures
from .table_join import TableJoiner
from .under_sampling import RandomUnderSampler
from .utils import get_categorical_feature_names
from .utils import get_multi_value_categorical_feature_names
from .utils import get_numerical_feature_names
from .utils import get_time_feature_names


class PipelineMaker(object):
    def __init__(
        self,
        info: Dict[str, Any],
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE],
        target_type: str,
        n_jobs: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        verbose: int = 1,
        # Parameters for a sampler
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        shuffle: bool = True,
        # Parameters for a multi-value categorical transformer
        lowercase: bool = True,
        n_features_per_column: int = 1_048_576,
        # Parameters for a numerical transformer
        max_iter: int = 10,
        # Parameters for a model
        metric: str = '',
        n_estimators: int = 100,
        # Parameters for hyperpermeter search
        cv: Union[int, BaseCrossValidator] = 5,
        max_depth: int = 7,
        n_trials: int = 10,
        scoring: Union[str, Callable[..., float]] = None,
        subsample: Union[int, float] = 1.0,
        timeout: float = None
    ) -> None:
        self.cv = cv
        self.info = info
        self.lowercase = lowercase
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.metric = metric
        self.n_estimators = n_estimators
        self.n_features_per_column = n_features_per_column
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.related_tables = related_tables
        self.sampling_strategy = sampling_strategy
        self.scoring = scoring
        self.shuffle = shuffle
        self.subsample = subsample
        self.target_type = target_type
        self.timeout = timeout
        self.verbose = verbose

    def make_joiner(self) -> BaseEstimator:
        return TableJoiner(
            self.info,
            self.related_tables,
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

    def make_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            DropInvariant(verbose=self.verbose),
            DropUniqueKey(verbose=self.verbose),
            # DropDuplicates(verbose=self.verbose),
            SimpleImputer(
                fill_value='missing',
                strategy='constant',
                verbose=self.verbose
            ),
            CountEncoder(
                dtype='float32',
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        )

    def make_multi_value_categorical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            SimpleImputer(
                fill_value='missing',
                strategy='constant',
                verbose=self.verbose
            ),
            MultiValueCategoricalVectorizer(
                dtype='float32',
                lowercase=self.lowercase,
                n_features_per_column=self.n_features_per_column,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        )

    def make_numerical_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            DropInvariant(verbose=self.verbose),
            DropCollinearFeatures(verbose=self.verbose),
            make_union(
                make_pipeline(
                    Clip(
                        dtype='float32',
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    StandardScaler(
                        dtype='float32',
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    ),
                    IterativeImputer(
                        estimator=LinearRegression(n_jobs=self.n_jobs),
                        max_iter=self.max_iter
                    ),
                    make_union(
                        PolynomialFeatures(
                            include_bias=False,
                            interaction_only=True
                        ),
                        SubtractedFeatures(
                            dtype='float32',
                            n_jobs=self.n_jobs,
                            verbose=self.verbose
                        )
                    )
                ),
                make_pipeline(
                    SimpleImputer(
                        fill_value=np.finfo('float32').max,
                        strategy='constant',
                        verbose=self.verbose
                    ),
                    CountEncoder(
                        dtype='float32',
                        n_jobs=self.n_jobs,
                        verbose=self.verbose
                    )
                ),
                MissingIndicator(error_on_new=False),
                RowStatistics(
                    dtype='float32',
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
            )
        )

    def make_time_transformer(self) -> BaseEstimator:
        return make_pipeline(
            NAProportionThreshold(verbose=self.verbose),
            SimpleImputer(strategy='min', verbose=self.verbose),
            SubtractedFeatures(
                dtype='float32',
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
        )

    def make_engineer(self) -> BaseEstimator:
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

    def make_model(self) -> BaseEstimator:
        params = {
            'max_depth': self.max_depth,
            'metric': self.metric,
            'n_estimators': self.n_estimators,
            'n_jobs': 1,
            'random_state': self.random_state,
            'subsample_freq': 1
        }

        if self.target_type in ['binary', 'multiclass', 'multiclass-output']:
            selector = SelectFpr()
            model = lgb.LGBMClassifier(**params)
        elif self.target_type in ['continuous', 'continuous-output']:
            selector = SelectFpr(score_func=f_regression)
            model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f'Unknown target_type: {self.target_type}.')

        return make_pipeline(
            selector,
            model
        )

    def make_search_cv(self) -> BaseEstimator:
        model = self.make_model()
        model_name = model._final_estimator.__class__.__name__.lower()
        param_distributions = {
            f'{model_name}__colsample_bytree':
                optuna.distributions.UniformDistribution(0.5, 1.0),
            f'{model_name}__learning_rate':
                optuna.distributions.LogUniformDistribution(0.001, 0.1),
            f'{model_name}__min_child_samples':
                optuna.distributions.IntUniformDistribution(1, 100),
            f'{model_name}__num_leaves':
                optuna.distributions.IntUniformDistribution(
                    2,
                    2 ** self.max_depth - 1
                ),
            f'{model_name}__reg_alpha':
                optuna.distributions.LogUniformDistribution(1e-06, 10.0),
            f'{model_name}__reg_lambda':
                optuna.distributions.LogUniformDistribution(1e-06, 10.0),
            f'{model_name}__subsample':
                optuna.distributions.UniformDistribution(0.5, 1.0)
        }

        return OptunaSearchCV(
            model,
            param_distributions,
            cv=self.cv,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            scoring=self.scoring,
            subsample=self.subsample,
            timeout=self.timeout,
            verbose=self.verbose
        )
