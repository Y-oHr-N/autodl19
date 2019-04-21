import lightgbm as lgb
import optuna

from category_encoders import OrdinalEncoder
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from .model_selection import OptunaSearchCV
from .preprocessing import Clip
from .utils import get_categorical_columns
from .utils import get_numerical_columns


def make_categorical_transformer() -> Pipeline:
    return Pipeline([
        (
            'imputer',
            SimpleImputer(fill_value='missing', strategy='constant')
        ),
        (
            'transformer',
            OrdinalEncoder()
        )
    ])


def make_numerical_transformer() -> Pipeline:
    return Pipeline([
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'transformer',
            Clip()
        )
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
            )
        ],
        n_jobs=4
    )


def make_optuna_search_cv() -> OptunaSearchCV:
    estimator = lgb.LGBMClassifier(
        boosting_type='gbdt',
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
        n_trials=20,
        n_jobs=4,
        sampler=sampler,
        scoring='roc_auc'
    )


def make_model() -> Pipeline:
    return Pipeline([
        ('under_sampler', RandomUnderSampler(random_state=0)),
        ('optuna_search_cv', make_optuna_search_cv())
    ])
