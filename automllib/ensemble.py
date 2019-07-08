from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Sequence
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight

from .base import BaseEstimator
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE

CLASSIFICATION_METRICS = {
    'binary': 'binary_logloss',
    'multiclass': 'multi_logloss',
    'softmax': 'multi_logloss',
    'multiclassova': 'multi_logloss',
    'multiclass_ova': 'multi_logloss',
    'ova': 'multi_logloss',
    'ovr': 'multi_logloss'
}
REGRESSION_METRICS = {
    'mean_absoluter_error': 'l1',
    'mae': 'l1',
    'regression_l1': 'l1',
    'l2_root': 'l2',
    'mean_squared_error': 'l2',
    'mse': 'l2',
    'regression': 'l2',
    'regression_l2': 'l2',
    'root_mean_squared_error': 'l2',
    'rmse': 'l2',
    'huber': 'huber',
    'fair': 'fair',
    'poisson': 'poisson',
    'quantile': 'quantile',
    'mean_absolute_percentage_error': 'mape',
    'mape': 'mape',
    'gamma': 'gamma',
    'tweedie': 'tweedie'
}
METRICS = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS}


class EnvExtractionCallback(object):
    @property
    def best_iteration_(self) -> int:
        return self._env.iteration + 1

    def __call__(self, env: NamedTuple) -> None:
        self._env = env


class Objective(object):
    _colsample_bytree_low = 0.1
    _colsample_bytree_high = 1.0
    _min_child_samples_low = 1
    _min_child_samples_high = 100
    _min_child_weight_low = 1e-03
    _min_child_weight_high = 10.0
    _num_leaves_low = 2
    _num_leaves_high = 127
    _reg_alpha_low = 1e-06
    _reg_alpha_high = 10.0
    _reg_lambda_low = 1e-06
    _reg_lambda_high = 10.0
    _subsample_low = 0.1
    _subsample_high = 1.0

    def __init__(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        params: Dict[str, Any],
        categorical_features: Union[Sequence[Union[int, str]], str] = 'auto',
        cv: BaseCrossValidator = None,
        enable_pruning: bool = False,
        n_estimators: int = 100,
        n_iter_no_change: int = None,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None,
        seed: int = 0
    ) -> None:
        self.categorical_features = categorical_features
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.params = params
        self.sample_weight = sample_weight
        self.seed = seed
        self.X = X
        self.y = y

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = {
            'colsample_bytree':
                trial.suggest_uniform(
                    'colsample_bytree',
                    self._colsample_bytree_low,
                    self._colsample_bytree_high
                ),
            'min_child_samples':
                trial.suggest_int(
                    'min_child_samples',
                    self._min_child_samples_low,
                    self._min_child_samples_high
                ),
            'min_child_weight':
                trial.suggest_loguniform(
                    'min_child_weight',
                    self._min_child_weight_low,
                    self._min_child_weight_high
                ),
            'num_leaves':
                trial.suggest_int(
                    'num_leaves',
                    self._num_leaves_low,
                    self._num_leaves_high
                ),
            'reg_alpha':
                trial.suggest_loguniform(
                    'reg_alpha',
                    self._reg_alpha_low,
                    self._reg_alpha_high
                ),
            'reg_lambda':
                trial.suggest_loguniform(
                    'reg_lambda',
                    self._reg_lambda_low,
                    self._reg_alpha_high
                ),
            'subsample':
                trial.suggest_uniform(
                    'subsample',
                    self._subsample_low,
                    self._subsample_high
                )
        }
        extraction_callback = EnvExtractionCallback()
        callbacks = [extraction_callback]

        if self.enable_pruning:
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial,
                self.params['metric']
            )

            callbacks.append(pruning_callback)

        params.update(self.params)

        dataset = lgb.Dataset(
            self.X,
            categorical_feature=self.categorical_features,
            label=self.y,
            params=params,
            weight=self.sample_weight
        )
        eval_hist = lgb.cv(
            params,
            dataset,
            callbacks=callbacks,
            early_stopping_rounds=self.n_iter_no_change,
            folds=self.cv,
            num_boost_round=self.n_estimators,
            seed=self.seed
        )

        if self.n_iter_no_change is None:
            best_iteration = None
        else:
            best_iteration = extraction_callback.best_iteration_

        trial.set_user_attr('best_iteration', best_iteration)

        return eval_hist[f"{self.params['metric']}-mean"][-1]


class BaseLGBMModelCV(BaseEstimator):
    # TODO(Kon): Add `groups` into fit
    # TODO(Kon): Search best `boosting_type`
    # TODO(Kon): Search best `min_split_gain`
    # TODO(Kon): Output SHAP values

    @property
    def _categorical_features(self) -> Union[Sequence[Union[int, str]], str]:
        if self.categorical_features is None:
            return 'auto'

        return self.categorical_features

    @property
    def best_index_(self) -> int:
        df = self.trials_dataframe()

        return df['value'].idxmin()

    @property
    def best_trial_(self) -> optuna.structs.FrozenTrial:
        self._check_is_fitted()

        return self.study_.best_trial

    @property
    def classes_(self) -> Sequence:
        self._check_is_fitted()

        return self.encoder_.classes_

    @property
    def feature_importances_(self) -> Sequence[float]:
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
    def trials_(self) -> Sequence[optuna.structs.FrozenTrial]:
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
        categorical_features: Union[Sequence[Union[int, str]], str] = None,
        class_weight: Union[str, Dict[str, float]] = None,
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = False,
        importance_type: str = 'split',
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        n_iter_no_change: int = None,
        n_jobs: int = 1,
        n_seeds: int = 10,
        n_trials: int = 10,
        objective: str = None,
        random_state: Union[int, np.random.RandomState] = None,
        study: optuna.study.Study = None,
        timeout: float = None,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.categorical_features = categorical_features
        self.class_weight = class_weight
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.importance_type = importance_type
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.n_seeds = n_seeds
        self.n_trials = n_trials
        self.objective = objective
        self.random_state = random_state
        self.study = study
        self.timeout = timeout

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None,
    ) -> 'BaseLGBMModelCV':
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, np.iinfo('int32').max)
        params = {
            'learning_rate': self.learning_rate,
            'n_jobs': 1,
            'seed': seed,
            'verbose': -1
        }
        is_classifier = self._estimator_type == 'classifier'
        cv = check_cv(self.cv, y, is_classifier)
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(self._parallel_fit_booster)
        logger = self._get_logger()

        if is_classifier:
            self.encoder_ = LabelEncoder()

            y = self.encoder_.fit_transform(y)

            self.n_classes_ = len(self.encoder_.classes_)

            if self.n_classes_ > 2:
                params['num_classes'] = self.n_classes_

                if self.objective is None:
                    params['objective'] = 'multiclass'

            else:
                if self.objective is None:
                    params['objective'] = 'binary'

        else:
            if self.objective is None:
                params['objective'] = 'regression'

        if self.objective is not None:
            params['objective'] = self.objective

        params['metric'] = METRICS[params['objective']]

        if self.class_weight is not None:
            if sample_weight is None:
                sample_weight = compute_sample_weight(self.class_weight, y)
            else:
                sample_weight *= compute_sample_weight(self.class_weight, y)

        objective = Objective(
            X,
            y,
            params,
            categorical_features=self._categorical_features,
            cv=cv,
            enable_pruning=self.enable_pruning,
            n_estimators=self.n_estimators,
            n_iter_no_change=self.n_iter_no_change,
            sample_weight=sample_weight
        )

        if self.study is None:
            pruner = optuna.pruners.SuccessiveHalvingPruner()
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(pruner=pruner, sampler=sampler)

        else:
            self.study_ = self.study

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        self.best_iteration_ = \
            self.study_.best_trial.user_attrs['best_iteration']
        self.best_params_ = {**params, **self.study_.best_params}
        self.best_score_ = self.study_.best_value

        logger.info(f'The best iteration is {self.best_iteration_}.')
        logger.info(f'The best CV score is {self.best_score_:.3f}.')

        if self.n_iter_no_change is None:
            n_estimators = self.n_estimators
        else:
            n_estimators = self.best_iteration_

        self.boosters_ = parallel(
            func(
                X,
                y,
                self.best_params_,
                n_estimators,
                sample_weight,
                random_state.randint(0, np.iinfo('int32').max)
            ) for _ in range(self.n_seeds)
        )

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'non_deterministic': True, 'no_validation': True}

    def _parallel_fit_booster(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        params: Dict[str, Any],
        n_estimators: int,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE,
        seed: int
    ) -> lgb.Booster:
        params = params.copy()
        params['seed'] = seed

        dataset = lgb.Dataset(
            X,
            categorical_feature=self._categorical_features,
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

    def _check_params(self) -> None:
        if self.objective is not None \
                and self.objective not in CLASSIFICATION_METRICS:
            raise ValueError(f'Invalid objective: {self.objective}')

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the Fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """

        probas = self.predict_proba(X)
        class_index = np.argmax(probas, axis=1)

        return self.encoder_.inverse_transform(class_index)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        """Predict class probabilities for data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        p
            Class probabilities of data.
        """

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

    def _check_params(self) -> None:
        if self.objective is not None \
                and self.objective not in REGRESSION_METRICS:
            raise ValueError(f'Invalid objective: {self.objective}')

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the Fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """

        self._check_is_fitted()

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.predict)
        results = parallel(func(b, X) for b in self.boosters_)

        return np.average(results, axis=0)
