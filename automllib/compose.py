import lightgbm as lgb
import optuna

from category_encoders import OrdinalEncoder
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.compose import make_column_transformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_union
from sklearn.preprocessing import PolynomialFeatures

from .feature_extraction import MultiValueCategoricalVectorizer
from .feature_extraction import TimeVectorizer
from .feature_selection import DropDuplicates
from .feature_selection import DropUniqueKey
from .feature_selection import NAProportionThreshold
from .feature_selection import NUniqueThreshold
from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .utils import get_categorical_feature_names
from .utils import get_multi_value_categorical_feature_names
from .utils import get_numerical_feature_names
from .utils import get_time_feature_names


def make_categorical_transformer(timeout: float = None) -> BaseEstimator:
    return make_pipeline(
        DropUniqueKey(),
        SimpleImputer(fill_value='missing', strategy='constant'),
        OrdinalEncoder()
    )


def make_multi_value_categorical_transformer(
    timeout: float = None
) -> BaseEstimator:
    return make_pipeline(
        SimpleImputer(fill_value='missing', strategy='constant'),
        MultiValueCategoricalVectorizer(),
        TruncatedSVD(n_components=5)
    )


def make_numerical_transformer(timeout: float = None) -> BaseEstimator:
    return make_union(
        make_pipeline(
            SimpleImputer(strategy='median'),
            Clip(copy=False),
            PolynomialFeatures(include_bias=False, interaction_only=True)
        ),
        MissingIndicator()
    )


def make_time_transformer(timeout: float = None) -> BaseEstimator:
    return make_pipeline(
        TimeVectorizer(),
        SimpleImputer(strategy='most_frequent')
    )


def make_mixed_transformer(timeout: float = None) -> BaseEstimator:
    return make_column_transformer(
        (
            make_categorical_transformer(timeout=timeout),
            get_categorical_feature_names
        ),
        (
            make_multi_value_categorical_transformer(timeout=timeout),
            get_multi_value_categorical_feature_names
        ),
        (
            make_numerical_transformer(timeout=timeout),
            get_numerical_feature_names
        ),
        (
            make_time_transformer(timeout=timeout),
            get_time_feature_names
        ),
        n_jobs=-1
    )


def make_preprocessor(timeout: float = None) -> BaseEstimator:
    return make_pipeline(
        NAProportionThreshold(),
        NUniqueThreshold(),
        DropDuplicates(),
        make_mixed_transformer(timeout=timeout)
    )


def make_search_cv(timeout: float = None) -> BaseEstimator:
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
        error_score='raise',
        n_trials=20,
        n_jobs=4,
        random_state=0,
        sampler=sampler,
        scoring='roc_auc',
        subsample=100_000,
        timeout=timeout
    )


def make_model(timeout: float = None) -> BaseEstimator:
    return make_pipeline(
        RandomUnderSampler(random_state=0),
        make_search_cv(timeout=timeout)
    )
