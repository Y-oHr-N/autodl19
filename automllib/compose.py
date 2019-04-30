import lightgbm as lgb
import optuna

from category_encoders import OrdinalEncoder
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion

from .feature_extraction import TimeVectorizer
from .feature_selection import DropDuplicates
from .feature_selection import DropUniqueKey
from .feature_selection import NAProportionThreshold
from .feature_selection import NUniqueThreshold
from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .utils import get_categorical_columns
from .utils import get_numerical_columns
from .utils import get_time_columns


def make_categorical_transformer(timeout: float = None) -> BaseEstimator:
    return Pipeline([
        ('categorical_selector', DropUniqueKey()),
        (
            'categorical_imputer',
            SimpleImputer(fill_value='missing', strategy='constant')
        ),
        ('categorical_transformer', OrdinalEncoder())
    ])


def make_numerical_transformer(timeout: float = None) -> BaseEstimator:
    return FeatureUnion([
        (
            'numerical_transformer',
            Pipeline([
                ('numerical_imputer', SimpleImputer(strategy='median')),
                ('numerical_transformer', Clip(copy=False))
            ])
        ),
        ('numerical_indicator', MissingIndicator())
    ])


def make_time_transformer(timeout: float = None) -> BaseEstimator:
    return Pipeline([
        ('time_vectorizer', TimeVectorizer()),
        ('time_imputer', SimpleImputer(strategy='most_frequent'))
    ])


def make_mixed_transformer(timeout: float = None) -> BaseEstimator:
    return ColumnTransformer(
        [
            (
                'categorical_transformer',
                make_categorical_transformer(timeout=timeout),
                get_categorical_columns
            ),
            (
                'numerical_transformer',
                make_numerical_transformer(timeout=timeout),
                get_numerical_columns
            ),
            (
                'time_transformer',
                make_time_transformer(timeout=timeout),
                get_time_columns
            )
        ],
        n_jobs=-1
    )


def make_preprocessor(timeout: float = None) -> BaseEstimator:
    return Pipeline([
        ('first_mixed_selector', NAProportionThreshold()),
        ('second_mixed_selector', NUniqueThreshold()),
        ('third_mixed_selector', DropDuplicates()),
        ('mixed_transformer', make_mixed_transformer(timeout=timeout))
    ])


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
    return Pipeline([
        ('sampler', RandomUnderSampler(random_state=0)),
        ('search_cv', make_search_cv(timeout=timeout))
    ])
