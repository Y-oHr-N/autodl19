from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna

from joblib import delayed
from joblib import effective_n_jobs
from joblib import Parallel
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class EnvExtractionCallback(object):
    @property
    def best_iteration_(self) -> int:
        return self._env.iteration + 1

    def __call__(self, env: NamedTuple) -> None:
        self._env = env


class Objective(object):
    def __init__(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        params: Dict[str, Any],
        categorical_feature: Union[str, List[Union[int, str]]] = 'auto',
        cv: BaseCrossValidator = None,
        metric: str = 'l2',
        n_estimators: int = 100,
        n_iter_no_change: int = None,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None,
        seed: int = 0
    ) -> None:
        self.categorical_feature = categorical_feature
        self.cv = cv
        self.metric = metric
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.params = params
        self.sample_weight = sample_weight
        self.seed = seed
        self.X = X
        self.y = y

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = self.params.copy()
        other_params = {
            'colsample_bytree':
                trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
            'min_child_samples':
                trial.suggest_int('min_child_samples', 1, 100),
            'min_child_weight':
                trial.suggest_loguniform('min_child_weight', 1e-03, 10.0),
            'num_leaves':
                trial.suggest_int('num_leaves', 2, 127),
            'reg_alpha':
                trial.suggest_loguniform('reg_alpha', 1e-06, 10.0),
            'reg_lambda':
                trial.suggest_loguniform('reg_lambda', 1e-06, 10.0),
            'subsample':
                trial.suggest_uniform('subsample', 0.1, 1.0)
        }
        extraction_callback = EnvExtractionCallback()
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial,
            self.metric
        )

        params.update(other_params)

        dataset = lgb.Dataset(
            self.X,
            categorical_feature=self.categorical_feature,
            label=self.y,
            params=params,
            weight=self.sample_weight
        )
        eval_hist = lgb.cv(
            params,
            dataset,
            callbacks=[extraction_callback, pruning_callback],
            early_stopping_rounds=self.n_iter_no_change,
            folds=self.cv,
            metrics=self.metric,
            num_boost_round=self.n_estimators,
            seed=self.seed
        )

        if self.n_iter_no_change is None:
            best_iteration = None
        else:
            best_iteration = extraction_callback.best_iteration_

        trial.set_user_attr('best_iteration', best_iteration)

        return eval_hist[f'{self.metric}-mean'][-1]


class BaseLGBMModelCV(BaseEstimator):
    # TODO(Kon): Add `groups` into fit
    # TODO(Kon): Search best `boosting_type`
    # TODO(Kon): Search best `min_split_gain`
    # TODO(Kon): Output SHAP values

    @property
    def best_index_(self) -> int:
        df = self.trials_dataframe()

        return df['value'].idxmin()

    @property
    def best_iteration_(self) -> int:
        return self.best_trial_.user_attrs['best_iteration']

    @property
    def best_params_(self) -> Dict[str, Any]:
        self._check_is_fitted()

        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        self._check_is_fitted()

        return self.study_.best_value

    @property
    def best_trial_(self) -> optuna.structs.FrozenTrial:
        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def feature_importances_(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.feature_importance)
        results = parallel(
            func(b, self.importance_type) for b in self.boosters_
        )

        return np.average(results, axis=0)

    @property
    def n_trials_(self) -> int:
        return len(self.trials_)

    @property
    def trials_(self) -> List[optuna.structs.FrozenTrial]:
        self._check_is_fitted()

        return self.study_.trials

    @property
    def user_attrs_(self) -> Dict[str, Any]:
        self._check_is_fitted()

        return self.study_.user_attrs

    @property
    def set_user_attr(self) -> Callable:
        self._check_is_fitted()

        return self.study_.set_user_attr

    @property
    def trials_dataframe(self) -> Callable:
        self._check_is_fitted()

        return self.study_.trials_dataframe

    def __init__(
        self,
        class_weight: Union[str, Dict[str, float]] = None,
        cv: Union[int, BaseCrossValidator] = 5,
        importance_type: str = 'split',
        learning_rate: float = 0.1,
        min_split_gain: float = 0.0,
        n_estimators: int = 1_000,
        n_iter_no_change: int = None,
        n_jobs: int = 1,
        n_seeds: int = 10,
        n_trials: int = 10,
        random_state: Union[int, np.random.RandomState] = None,
        study: optuna.study.Study = None,
        subsample_for_bin: int = 200_000,
        timeout: float = None,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)

        self.class_weight = class_weight
        self.cv = cv
        self.importance_type = importance_type
        self.learning_rate = learning_rate
        self.min_split_gain = min_split_gain
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.random_state = random_state
        self.study = study
        self.subsample_for_bin = subsample_for_bin
        self.timeout = timeout

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None,
        categorical_feature: Union[str, List[Union[int, str]]] = 'auto'
    ) -> 'BaseLGBMModelCV':
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo('int32').max)
        params = {
            'learning_rate': self.learning_rate,
            'min_split_gain': self.min_split_gain,
            'n_jobs': 1,
            'seed': seed,
            'subsample_for_bin': self.subsample_for_bin,
            'verbose': -1
        }
        classifier = self._estimator_type == 'classifier'
        cv = check_cv(self.cv, y, classifier)

        if classifier:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            if self.n_classes_ > 2:
                direction = 'minimize'
                metric = 'multi_logloss'
                params['num_classes'] = self.n_classes_
                params['objective'] = 'multiclass'
            else:
                direction = 'maximize'
                metric = 'auc'
                params['is_unbalance'] = True
                params['objective'] = 'binary'

        else:
            direction = 'minimize'
            metric = 'l2'
            params['objective'] = 'regression'

        if self.class_weight is not None:
            if sample_weight is None:
                sample_weight = compute_sample_weight(self.class_weight, y)
            else:
                sample_weight *= compute_sample_weight(self.class_weight, y)

        func = Objective(
            X,
            y,
            params,
            categorical_feature=categorical_feature,
            cv=cv,
            metric=metric,
            n_estimators=self.n_estimators,
            n_iter_no_change=self.n_iter_no_change,
            sample_weight=sample_weight
        )

        if self.study is None:
            pruner = optuna.pruners.SuccessiveHalvingPruner()
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(
                direction=direction,
                pruner=pruner,
                sampler=sampler
            )

        else:
            self.study_ = self.study

        self.study_.optimize(
            func,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        logger = self._get_logger()
        best_iteration = self.study_.best_trial.user_attrs['best_iteration']
        best_score = self.study_.best_value

        logger.info(f'The best iteration is {best_iteration}.')
        logger.info(f'The best CV score is {best_score:.3f}.')

        params.update(self.study_.best_params)

        if self.n_iter_no_change is None:
            n_estimators = self.n_estimators
        else:
            n_estimators = best_iteration

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(self._parallel_refit_booster)

        self.boosters_ = parallel(
            func(
                X,
                y,
                params,
                categorical_feature,
                n_estimators,
                random_state,
                sample_weight
            ) for _ in range(self.n_seeds)
        )

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}

    def _parallel_refit_booster(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        params: Dict[str, Any],
        categorical_feature: Union[str, List[Union[int, str]]],
        n_estimators: int,
        random_state: np.random.RandomState,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE
    ) -> lgb.Booster:
        params = params.copy()

        if self.n_seeds > 1:
            seed = random_state.randint(0, np.iinfo('int32').max)
            params['seed'] = seed

        dataset = lgb.Dataset(
            X,
            categorical_feature=categorical_feature,
            label=y,
            params=params,
            weight=sample_weight
        )
        booster = lgb.train(
            params,
            dataset,
            num_boost_round=n_estimators
        )

        booster.free_dataset()

        return booster

    @abstractmethod
    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        pass


class LGBMClassifierCV(BaseLGBMModelCV, ClassifierMixin):
    """

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from automllib.ensemble import LGBMClassifierCV
    >>> clf = LGBMClassifierCV(n_iter_no_change=10, random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    LGBMClassifierCV(...)
    >>> clf.score(X, y)
    0.9...
    """

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        probas = self.predict_proba(X)

        return self.classes_[np.argmax(probas, axis=1)]

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.predict)
        results = parallel(func(b, X) for b in self.boosters_)
        result = np.average(results, axis=0)

        if self.n_classes_ > 2:
            return result

        else:
            result = result.reshape(-1, 1)

            return np.concatenate([1.0 - result, result], axis=1)


class LGBMRegressorCV(BaseLGBMModelCV, RegressorMixin):
    """

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from automllib.ensemble import LGBMRegressorCV
    >>> reg = LGBMRegressorCV(n_iter_no_change=10, random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    LGBMRegressorCV(...)
    >>> reg.score(X, y)
    0.9...
    """

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        self._check_is_fitted()

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.predict)
        results = parallel(func(b, X) for b in self.boosters_)

        return np.average(results, axis=0)
