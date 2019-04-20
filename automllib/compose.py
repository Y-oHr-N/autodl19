import lightgbm as lgb
import optuna

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from .model_selection import OptunaSearchCV
from .preprocessing import Clip


def make_model() -> Pipeline:
    return Pipeline([
        ('transformer', Clip()),
        ('under_sampler', RandomUnderSampler(random_state=0)),
        ('optuna_search_cv', make_optuna_search_cv())
    ])


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
