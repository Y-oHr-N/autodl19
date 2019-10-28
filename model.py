import logging
import os
import pickle
import time

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q git+https://github.com/Y-oHr-N/OptGBM.git')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas')
os.system('pip3 install -q scikit-learn')

import colorlog
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMClassifier
from sklearn.model_selection import check_cv
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_random_state

from automllib.utils import Timeit
from automllib.utils import Timer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s'
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

timeit = Timeit(logger=logger)


CATEGORICAL_PREFIX = 'c_'
MULTI_VALUE_CATEGORICAL_PREFIX = 'm_'
NUMERICAL_PREFIX = 'n_'
TIME_PREFIX = 't_'


class Enginner(object):
    def __init__(self, high=99.9, low=0.01):
        self.high = high
        self.low = low

    @timeit
    def fit(self, X):
        self.numerical_features_ = \
            [c for c in X if c.startswith(NUMERICAL_PREFIX)]
        self.categorical_features_ = \
            [c for c in X if c.startswith(CATEGORICAL_PREFIX)]
        self.multi_value_categorical_features_ = \
            [c for c in X if c.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]
        self.time_features_ = [c for c in X if c.startswith(TIME_PREFIX)]

        if len(self.numerical_features_) > 0:
            self.data_min_, self.data_max_ = np.nanpercentile(
                X[self.numerical_features_],
                [self.low, self.high],
                axis=0
            )

        self.frequency_ = X.nunique()

        logger.info(self.frequency_)

        return self

    @timeit
    def transform(self, X):
        if len(self.categorical_features_) > 0:
            X[self.categorical_features_] = \
                X[self.categorical_features_].astype('category')

        if len(self.multi_value_categorical_features_) > 0:
            X[self.multi_value_categorical_features_] = \
                X[self.multi_value_categorical_features_].apply(
                    lambda x: np.nan if np.isnan(x) else hash(x)
                )
            X[self.multi_value_categorical_features_] = \
                X[self.multi_value_categorical_features_].astype('category')

        if len(self.numerical_features_) > 0:
            X[self.numerical_features_] = \
                X[self.numerical_features_].clip(
                    self.data_min_,
                    self.data_max_,
                    axis=1
                )
            X[self.numerical_features_] = \
                X[self.numerical_features_].astype('float32')

        dropped_features = X.columns[self.frequency_ == 1]

        if len(self.time_features_) > 0:
            dropped_features = dropped_features.union(self.time_features_)

        X = X.drop(columns=dropped_features)

        return X


class AutoSSLClassifier(object):
    @property
    def predict_proba(self):
        return self.model_.predict_proba

    def __init__(
        self,
        cv=5,
        high=95.0,
        low=5.0,
        max_samples=100_000,
        n_jobs=1,
        n_trials=25,
        random_state=None,
        timeout=None
    ):
        self.cv = cv
        self.high = high
        self.low = low
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.timeout = timeout

        self._timer = Timer(time_budget=timeout)

        self._timer.start()

    def fit(self, X, y, **fit_params):
        n_samples, _ = X.shape
        is_labeled = y != 0
        X_labeled = X[is_labeled]
        y_labeled = y[is_labeled]
        iter_time = 0.0

        while self._timer.get_remaining_time() - iter_time > 0:
            start_time = time.perf_counter()

            self.model_ = AutoNoisyClassifier(
                cv=self.cv,
                max_samples=self.max_samples,
                n_jobs=self.n_jobs,
                n_trials=100,
                random_state=self.random_state,
                timeout=None
            )

            self.model_.fit(X_labeled, y_labeled, **fit_params)

            if is_labeled.sum() == n_samples:
                break

            y_score = np.full(n_samples, np.nan)
            y_score[~is_labeled] = self.model_.predict_proba(
                X[~is_labeled]
            )[:, 1]

            low_value, high_value = np.nanpercentile(
                y_score,
                [self.low, self.high]
            )
            is_high = y_score >= high_value
            is_low = y_score <= low_value
            y[~is_labeled & is_high] = 1
            y[~is_labeled & is_low] = -1
            is_labeled |= is_high
            is_labeled |= is_low

            X_labeled = X[is_labeled]
            y_labeled = y[is_labeled]

            iter_time = time.perf_counter() - start_time

        return self


class AutoPUClassifier(object):
    def __init__(
        self,
        cv=5,
        max_samples=100_000,
        n_iter=10,
        n_jobs=1,
        n_trials=25,
        random_state=None,
        timeout=None
    ):
        self.cv = cv
        self.max_samples = max_samples
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.timeout = timeout

        self._timer = Timer(time_budget=timeout)

        self._timer.start()

    def fit(self, X, y, **fit_params):
        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape
        n_pos_samples = np.sum(y == 1)
        sample_indices = np.arange(n_samples)
        sample_indices_positive = sample_indices[y == 1]
        timeout = self._timer.get_remaining_time() / self.n_iter

        self.models_ = []

        for _ in range(self.n_iter):
            sample_indices_unlabeled = random_state.choice(
                sample_indices[y == 0],
                n_pos_samples,
                replace=False
            )
            sample_indices_selected = np.union1d(
                sample_indices_positive,
                sample_indices_unlabeled
            )
            model = AutoNoisyClassifier(
                cv=self.cv,
                max_samples=self.max_samples,
                n_jobs=self.n_jobs,
                n_trials=None,
                random_state=self.random_state,
                timeout=timeout
            )

            model.fit(
                X.iloc[sample_indices_selected],
                y.iloc[sample_indices_selected],
                **fit_params
            )

            self.models_.append(model)

        return self

    def predict_proba(self, X):
        for i, model in enumerate(self.models_):
            p = model.predict_proba(X)

            if i == 0:
                probas = p
            else:
                probas += p

        return probas / self.n_iter


class AutoNoisyClassifier(object):
    @property
    def predict_proba(self):
        return self.model_.predict_proba

    def __init__(
        self,
        cv=5,
        max_samples=100_000,
        n_jobs=1,
        n_trials=25,
        random_state=None,
        timeout=None
    ):
        self.cv = cv
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.timeout = timeout

        self._timer = Timer(time_budget=timeout)

        self._timer.start()

    def fit(self, X, y, **fit_params):
        random_state = check_random_state(self.random_state)
        cv = check_cv(self.cv, y=y, classifier=True)
        n_samples, _ = X.shape

        if n_samples > self.max_samples:
            if isinstance(cv, TimeSeriesSplit):
                X = X.iloc[-self.max_samples:]
                y = y.iloc[-self.max_samples:]
            else:
                sample_indices = np.arange(n_samples)
                sample_indices = random_state.choice(
                    sample_indices,
                    self.max_samples,
                    replace=False
                )

                sample_indices.sort()

                X = X.iloc[sample_indices]
                y = y.iloc[sample_indices]

        if self.timeout is None:
            timeout = None
        else:
            timeout = self._timer.get_remaning_time()

        self.model_ = OGBMClassifier(
            cv=cv,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=random_state,
            timeout=timeout
        )

        self.model_.fit(X, y, **fit_params)

        return self


class Model(object):
    def __init__(
        self,
        info: dict,
        cv=5,
        max_samples=30_000,
        n_jobs=-1,
        n_trials=None,
        random_state=0
    ):
        self.cv = cv
        self.info = info
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state

        logger.info(f'info={info}')

        self._timer = Timer()

        self._timer.start()

    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        time_features = [c for c in X if c.startswith(TIME_PREFIX)]

        if len(time_features) > 0:
            cv = TimeSeriesSplit(self.cv)
        else:
            cv = self.cv

        for c in time_features:
            X = X.sort_values(c)
            y = y.loc[X.index]

        self.engineer_ = Enginner()

        self.engineer_.fit(X)

        X = self.engineer_.transform(X)

        logger.info(f'X.shape={X.shape}')

        if self.info['task'] == 'ssl':
            klass = AutoSSLClassifier
        elif self.info['task'] == 'pu':
            klass = AutoPUClassifier
        elif self.info['task'] == 'noisy':
            klass = AutoNoisyClassifier

        timeout = \
            0.8 * self.info['time_budget'] - self._timer.get_elapsed_time()

        self.model_ = klass(
            cv=cv,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=self.random_state,
            timeout=timeout
        )

        self.model_.fit(X, y, eval_metric='auc')

    @timeit
    def predict(self, X: pd.DataFrame):
        X = self.engineer_.transform(X)

        logger.info(f'X.shape={X.shape}')

        probas = self.model_.predict_proba(X)

        return pd.Series(probas[:, 1])

    @timeit
    def save(self, directory: str):
        with open(os.path.join(directory, 'engineer.pkl'), 'wb') as f:
            pickle.dump(self.engineer_, f)

        with open(os.path.join(directory, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model_, f)

    @timeit
    def load(self, directory: str):
        with open(os.path.join(directory, 'engineer.pkl'), 'rb') as f:
            self.engineer_ = pickle.load(f)

        with open(os.path.join(directory, 'model.pkl'), 'rb') as f:
            self.model_ = pickle.load(f)
