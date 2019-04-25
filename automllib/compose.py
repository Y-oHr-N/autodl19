import lightgbm as lgb
import optuna

from category_encoders import OrdinalEncoder
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion

from .feature_extraction import TimeVectorizer
from .feature_selection import NAProportionThreshold
from .feature_selection import NUniqueThreshold
from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .utils import get_categorical_columns
from .utils import get_numerical_columns
from .utils import get_time_columns


def make_categorical_transformer() -> Pipeline:
    return Pipeline([
        (
            'categorical_imputer',
            SimpleImputer(
                fill_value='missing',
                strategy='constant'
            )
        ),
        ('categorical_transformer', OrdinalEncoder())
    ])


def make_numerical_transformer() -> FeatureUnion:
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


def make_time_transformer() -> Pipeline:
    return Pipeline([
        ('time_vectorizer', TimeVectorizer()),
        ('time_imputer', SimpleImputer(strategy='most_frequent'))
    ])


def make_mixed_transformer() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                'categorical_transformer',
                make_categorical_transformer(),
                get_categorical_columns
            ),
            (
                'numerical_transformer',
                make_numerical_transformer(),
                get_numerical_columns
            ),
            (
                'time_transformer',
                make_time_transformer(),
                get_time_columns
            )
        ],
        n_jobs=-1
    )


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ('1st_selector', NAProportionThreshold()),
        ('2nd_selector', NUniqueThreshold()),
        ('mixed_transformer', make_mixed_transformer())
    ])


def make_search_cv() -> OptunaSearchCV:
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
        subsample=100_000
    )


def make_model() -> Pipeline:
    return Pipeline([
        ('sampler', RandomUnderSampler(random_state=0)),
        ('search_cv', make_search_cv())
    ])
