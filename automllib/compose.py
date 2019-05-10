import lightgbm as lgb
import optuna

from imblearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
# from sklearn.impute import IterativeImputer
# from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_union
# from sklearn.preprocessing import PolynomialFeatures

from .base import BaseEstimator
from .feature_extraction import MultiValueCategoricalVectorizer
from .feature_extraction import TimeVectorizer
# from .feature_selection import DropDuplicates
from .feature_selection import DropInvariant
from .feature_selection import DropUniqueKey
from .feature_selection import NAProportionThreshold
from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .preprocessing import CountEncoder
from .under_sampling import RandomUnderSampler
from .utils import get_categorical_feature_names
from .utils import get_multi_value_categorical_feature_names
from .utils import get_numerical_feature_names
# from .utils import get_time_feature_names


def make_categorical_transformer(verbose: int = 0) -> BaseEstimator:
    return make_pipeline(
        NAProportionThreshold(verbose=verbose),
        DropInvariant(verbose=verbose),
        DropUniqueKey(verbose=verbose),
        # DropDuplicates(verbose=verbose),
        SimpleImputer(fill_value='missing', strategy='constant'),
        CountEncoder(dtype='float32', n_jobs=-1, verbose=verbose)
    )


def make_multi_value_categorical_transformer(
    verbose: int = 0
) -> BaseEstimator:
    return make_pipeline(
        NAProportionThreshold(verbose=verbose),
        SimpleImputer(fill_value='missing', strategy='constant'),
        MultiValueCategoricalVectorizer(
            dtype='float32',
            lowercase=False,
            n_features_per_column=64,
            n_jobs=-1,
            verbose=verbose
        )
    )


def make_numerical_transformer(verbose: int = 0) -> BaseEstimator:
    return make_pipeline(
        NAProportionThreshold(verbose=verbose),
        DropInvariant(verbose=verbose),
        make_union(
            make_pipeline(
                # IterativeImputer(),
                Clip(dtype='float32', verbose=verbose),
                # PolynomialFeatures(include_bias=False, interaction_only=True)
            ),
            # MissingIndicator()
        )
    )


def make_time_transformer(verbose: int = 0) -> BaseEstimator:
    return make_pipeline(
        NAProportionThreshold(verbose=verbose),
        SimpleImputer(
            fill_value=np.datetime64('1970-01-01'),
            strategy='constant'
        ),
        TimeVectorizer(verbose=verbose)
    )


def make_mixed_transformer(verbose: int = 0) -> BaseEstimator:
    return make_column_transformer(
        (
            make_categorical_transformer(verbose=verbose),
            get_categorical_feature_names
        ),
        (
            make_multi_value_categorical_transformer(verbose=verbose),
            get_multi_value_categorical_feature_names
        ),
        (
            make_numerical_transformer(verbose=verbose),
            get_numerical_feature_names
        ),
        # (
        #     make_time_transformer(verbose=verbose),
        #     get_time_feature_names
        # ),
    )


def make_search_cv(timeout: float = None, verbose: int = 0) -> BaseEstimator:
    estimator = lgb.LGBMClassifier(
        max_depth=7,
        metric='auc',
        # n_estimators=1000,
        n_jobs=1,
        random_state=0,
        subsample_freq=1
    )

    param_distributions = {
        'colsample_bytree': optuna.distributions.UniformDistribution(
            low=0.5,
            high=1.0
        ),
        'learning_rate': optuna.distributions.LogUniformDistribution(
            low=0.001,
            high=0.1
        ),
        'min_child_samples': optuna.distributions.IntUniformDistribution(
            low=1,
            high=100
        ),
        'num_leaves': optuna.distributions.IntUniformDistribution(
            low=2,
            high=123
        ),
        'reg_alpha': optuna.distributions.LogUniformDistribution(
            low=1e-06,
            high=10.0
        ),
        'reg_lambda': optuna.distributions.LogUniformDistribution(
            low=1e-06,
            high=10.0
        ),
        'subsample': optuna.distributions.UniformDistribution(
            low=0.5,
            high=1.0
        )
    }

    sampler = optuna.samplers.TPESampler(seed=0)

    return OptunaSearchCV(
        estimator,
        param_distributions,
        cv=TimeSeriesSplit(5),
        n_jobs=-1,
        random_state=0,
        sampler=sampler,
        scoring='roc_auc',
        subsample=100_000,
        timeout=timeout
    )


def make_model(timeout: float = None, verbose: int = 0) -> BaseEstimator:
    return make_pipeline(
        RandomUnderSampler(random_state=0, shuffle=False, verbose=verbose),
        make_search_cv(timeout=timeout, verbose=verbose)
    )
