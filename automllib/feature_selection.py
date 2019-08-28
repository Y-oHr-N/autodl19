from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn.metrics
import random
import optuna

from scipy.sparse import issparse
from scipy.stats import ks_2samp
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

from .base import BaseSelector
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE

MAX_INT = np.iinfo(np.int32).max


class DropCollinearFeatures(BaseSelector):
    """Drop collinear features.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import DropCollinearFeatures
    >>> sel = DropCollinearFeatures()
    >>> X = [[1, 1, 100], [2, 2, 10], [1, 1, 1], [1, 1, np.nan]]
    >>> Xt = sel.fit_transform(X)
    >>> Xt.shape
    (4, 2)
    """

    def __init__(
        self,
        random_state: Union[int, np.random.RandomState] = None,
        subsample: Union[int, float] = 1.0,
        threshold: float = 0.95,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.random_state = random_state
        self.subsample = subsample
        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'DropCollinearFeatures':
        random_state = check_random_state(self.random_state)
        X = X.astype('float64')
        n_samples, _ = X.shape

        if isinstance(self.subsample, int):
            max_samples = self.subsample
        else:
            max_samples = int(self.subsample * n_samples)

        if max_samples < n_samples:
            indices = random_state.choice(
                n_samples,
                max_samples,
                replace=False
            )
            X = X[indices]

        self.corr_ = pd._libs.algos.nancorr(X)

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        triu = np.triu(self.corr_, k=1)
        triu = np.abs(triu)
        triu = np.nan_to_num(triu)

        return np.all(triu <= self.threshold, axis=0)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}


class DropDriftFeatures(BaseSelector):
    """Drop drift features.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import DropDriftFeatures
    >>> sel = DropDriftFeatures()
    >>> X = [[1, 1, 100], [2, 2, 10], [1, 1, 1], [np.nan, 1, 1]]
    >>> X_test = [[1, 1000, 100], [2, 300, 10], [1, 100, 1], [1, 100, 1]]
    >>> Xt = sel.fit_transform(X, X_test=X_test)
    >>> Xt.shape
    (4, 2)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_samples: int = 100_000,
        random_state: Union[int, np.random.RandomState] = None,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.alpha = alpha
        self.max_samples = max_samples
        self.random_state = random_state

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        X_test: TWO_DIM_ARRAYLIKE_TYPE = None
    ) -> 'DropDriftFeatures':
        if X_test is None:
            self.pvalues_ = None

            return self

        X_test, _ = self._check_X_y(X_test)
        random_state = check_random_state(self.random_state)
        train_size, _ = X.shape
        train_size = min(train_size, self.max_samples)
        test_size, _ = X_test.shape
        test_size = min(test_size, self.max_samples)

        self.pvalues_ = np.empty(self.n_features_)

        for j in range(self.n_features_):
            column = X[:, j]
            column_test = X_test[:, j]
            is_nan = pd.isnull(column)
            is_nan_test = pd.isnull(column_test)
            train = np.where(~is_nan)[0]
            train = random_state.choice(train, size=train_size)
            test = np.where(~is_nan_test)[0]
            test = random_state.choice(test, size=test_size)
            column = safe_indexing(column, train)
            column_test = safe_indexing(column_test, test)

            if issparse(column):
                column = np.ravel(column.toarray())

            if issparse(column_test):
                column_test = np.ravel(column_test.toarray())

            self.pvalues_[j] = ks_2samp(column, column_test).pvalue

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        if self.pvalues_ is None:
            return np.ones(self.n_features_, dtype=bool)

        return self.pvalues_ >= self.alpha

    def _more_tags(self) -> Dict[str, Any]:
        return {
            'allow_nan': True,
            'non_deterministic': True,
            'X_types': ['2darray', 'sparse']
        }


class FrequencyThreshold(BaseSelector):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import FrequencyThreshold
    >>> sel = FrequencyThreshold()
    >>> X = [[1, 1, 'Cat'], [2, 2, 'Cat'], [1, 3, 'Cat'], [1, 4, np.nan]]
    >>> Xt = sel.fit_transform(X)
    >>> Xt.shape
    (4, 1)
    """

    @property
    def _max_frequency(self) -> int:
        if self.max_frequency is None:
            return MAX_INT

        return self.max_frequency

    def __init__(
        self,
        max_frequency: Union[int, float, None] = 1.0,
        min_frequency: Union[int, float] = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.max_frequency = max_frequency
        self.min_frequency = min_frequency

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'FrequencyThreshold':
        self.n_samples_, _ = X.shape
        self.frequency_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        max_frequency = self._max_frequency
        min_frequency = self.min_frequency

        if isinstance(max_frequency, float):
            max_frequency = int(max_frequency * self.n_samples_)

        if isinstance(min_frequency, float):
            min_frequency = int(min_frequency * self.n_samples_)

        return (self.frequency_ > min_frequency) \
            & (self.frequency_ < max_frequency)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}


class NAProportionThreshold(BaseSelector):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import NAProportionThreshold
    >>> sel = NAProportionThreshold()
    >>> X = [[1, 1, 'Cat'], [2, 2, np.nan], [1, 1, np.nan], [1, 1, np.nan]]
    >>> Xt = sel.fit_transform(X)
    >>> Xt.shape
    (4, 2)
    """

    def __init__(self, threshold: float = 0.6, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'NAProportionThreshold':
        self.n_samples_, _ = X.shape
        self.count_ = np.array([pd.Series.count(column) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        return self.count_ >= (1.0 - self.threshold) * self.n_samples_

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}


class Objective(object):

    _max_depth_choices = (-1, 2, 3, 4, 5, 6, 7)
    _num_leaves_low = 10
    _num_leaves_high = 200
    #_feature_fraction_choices = (0.6, 0.7, 0.8, 0.9, 1.0)
    #_bagging_fraction_choices = (0.6, 0.7, 0.8, 0.9, 1.0)
    _bagging_freq_choices = (0, 10, 20, 30, 40, 50)
    _reg_alpha_low = 1e-06
    _reg_alpha_high = 2.0
    _reg_lambda_low = 1e-06
    _reg_lambda_high = 2.0
    _min_child_weight_low = 1e-03
    _min_child_weight_high = 10.0

    def __init__(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        val_X: TWO_DIM_ARRAYLIKE_TYPE,
        val_y: ONE_DIM_ARRAYLIKE_TYPE,
        params: Dict[str, Any],
        num_boost_round: int = None,
        early_stopping_rounds: int = None,
    ) -> None:
        self.X = X
        self.y = y
        self.val_X = val_X
        self.val_y = val_y
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def __call__(self, trial: optuna.trial.Trial) -> float:

        params = {
            'max_depth':
                trial.suggest_categorical(
                    'max_depth',
                    self._max_depth_choices
                ),
            'num_leaves':
                trial.suggest_int(
                    'num_leaves',
                    self._num_leaves_low,
                    self._num_leaves_high
                ),
            #'feature_fraction':
                #trial.suggest_categorical(
                    #"feature_fraction",
                    #self._feature_fraction_choices,
                #),
            #'bagging_fraction':
                #trial.suggest_categorical(
                    #'bagging_fraction',
                    #self._bagging_fraction_choices,
                #),
            'bagging_freq':
                trial.suggest_categorical(
                    'bagging_freq',
                    self._bagging_freq_choices,
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
                    self._reg_lambda_high
                ),
            'min_child_weight':
                    trial.suggest_loguniform(
                    'min_child_weight',
                    self._min_child_weight_low,
                    self._min_child_weight_high
                ),
        }

        params.update(self.params)

        train_data = lgb.Dataset(
            self.X,
            label=self.y,
            params=params,
        )

        val_data = lgb.Dataset(
            self.val_X,
            label=self.val_y,
            params=params,
            reference = train_data,
        )

        hyperopt_model = lgb.train(
            params,
            train_data,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            valid_sets=val_data
        )

        best_iteration = hyperopt_model.best_iteration
        trial.set_user_attr('best_iteration', best_iteration)

        score = hyperopt_model.best_score["valid_0"][params["metric"]]

        return score


class FeatureSelector(BaseSelector):

    def __init__(
        self,
        time_col: str = None,
        train_size: float = 0.8, # Use 80% data for training
        train_size_for_searching : float = 0.4, # Use 40% train data for tuning
        valid_size: float = 0.2, # Use 20% tuning data for validation
        learning_rate: float = 0.01,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
        n_trials: int = 2,
        importance_type: str = 'split',
        k: int = 0,
        study: optuna.study.Study = None,
        seed: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        self.time_col = time_col
        self.train_size = train_size
        self.train_size_for_searching = train_size_for_searching
        self.valid_size = valid_size
        self.learning_rate = learning_rate
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.n_trials = n_trials
        self.importance_type = importance_type
        self.k = k
        self.seed = seed
        self.study = study

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'SelectFeaturesLGBM':

        train_len = int(self.train_size * len(X))
        if train_len == 0:
            train_len = 1

        if self.time_col is None:
            train_X = X[random.sample(range(0,X.shape[0]),train_len)]
            train_y = y[random.sample(range(0,X.shape[0]),train_len)]
        else:
            train_X = X[:-train_len]
            train_y = y[:-train_len]

        tuning_len = int(self.train_size_for_searching * len(train_X))
        if tuning_len == 0:
            tuning_len = 1

        if self.time_col is None:
            tuning_X = train_X[random.sample(range(0,train_X.shape[0]),tuning_len)]
            tuning_y = train_y[random.sample(range(0,train_X.shape[0]),tuning_len)]
        else:
            tuning_X = train_X[:-tuning_len]
            tuning_y = train_y[:-tuning_len]

        val_len = int(self.valid_size * len(tuning_X))
        if val_len == 0:
            val_len = 1

        tuning_train_X = tuning_X[:-val_len]
        tuning_val_X = tuning_X[-val_len:]

        tuning_train_y = tuning_y[:-val_len]
        tuning_val_y = tuning_y[-val_len:]

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'verbose': 1,
        }

        objective = Objective(
            tuning_train_X,
            tuning_train_y,
            tuning_val_X,
            tuning_val_y,
            params,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        self.study_ = optuna.create_study()

        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
        )

        if(len(tuning_X) != 1):
            self.best_iteration_ = self.study_.best_trial.user_attrs['best_iteration']
            self.best_params_ = {**params, **self.study_.best_params}
        else:
            self.best_iteration_ = self.num_boost_round
            self.best_params_ = params

        train_data = lgb.Dataset(train_X, train_y)
        self.model_ = lgb.train(
            params = self.best_params_,
            num_boost_round = self.best_iteration_,
            train_set = train_data,
        )

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        importance_array = self.model_.feature_importance(importance_type=self.importance_type)
        importance_index = np.argsort(importance_array)

        if self.n_features_ < self.k:
            return importance_index >= 0
        else:
            return importance_index > self.n_features_ - self.k
