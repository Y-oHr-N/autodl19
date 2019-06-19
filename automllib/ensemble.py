import copy

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

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state

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
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        categorical_feature: Union[str, List[Union[int, str]]] = 'auto',
        cv: BaseCrossValidator = None,
        metric: str = 'l2',
        n_estimators: int = 100,
        n_iter_no_change: int = None,
        seed: int = 0
    ) -> None:
        self.categorical_feature = categorical_feature
        self.cv = cv
        self.dataset = dataset
        self.metric = metric
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.params = params
        self.seed = seed

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
        train_set = copy.copy(self.dataset)
        extraction_callback = EnvExtractionCallback()
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial,
            self.metric
        )

        params.update(other_params)

        eval_hist = lgb.cv(
            params,
            train_set,
            callbacks=[extraction_callback, pruning_callback],
            categorical_feature=self.categorical_feature,
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
    # TODO(Kon): Add `class_weight` into __init__
    # TODO(Kon): Add `sample_weight` into fit
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

        results = []

        for b in self.boosters_:
            result = b.feature_importance(self.importance_type)

            results.append(result)

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
        categorical_feature: Union[str, List[Union[int, str]]] = 'auto',
        cv: Union[int, BaseCrossValidator] = 5,
        importance_type: str = 'split',
        learning_rate: float = 0.1,
        min_split_gain: float = 0.0,
        n_estimators: int = 100,
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

        self.categorical_feature = categorical_feature
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
        y: ONE_DIM_ARRAYLIKE_TYPE
    ) -> 'LGBMModelCV':
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
        dataset = lgb.Dataset(X, label=y)
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

        func = Objective(
            params,
            dataset,
            categorical_feature=self.categorical_feature,
            cv=cv,
            metric=metric,
            n_estimators=self.n_estimators,
            n_iter_no_change=self.n_iter_no_change
        )

        if self.study is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(
                direction=direction,
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

        params['n_jobs'] = self.n_jobs

        params.update(self.study_.best_params)

        if self.n_iter_no_change is None:
            num_boost_round = self.n_estimators
        else:
            num_boost_round = \
                self.study_.best_trial.user_attrs['best_iteration']

        self.boosters_ = []

        for i in range(self.n_seeds):
            if self.n_seeds > 1:
                seed = random_state.randint(0, np.iinfo('int32').max)
                params['seed'] = seed

            train_set = copy.copy(dataset)
            b = lgb.train(
                params,
                train_set,
                categorical_feature=self.categorical_feature,
                num_boost_round=num_boost_round
            )

            b.free_dataset()

            self.boosters_.append(b)

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}

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

        results = []

        for b in self.boosters_:
           result = b.predict(X)

           results.append(result)

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

        results = []

        for b in self.boosters_:
           result = b.predict(X)

           results.append(result)

        return np.average(results, axis=0)
