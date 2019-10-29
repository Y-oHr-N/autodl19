import itertools
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
from sklearn.base import BaseEstimator
from sklearn.base import clone
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


class Enginner(BaseEstimator):
    def __init__(
        self,
        high=99.0,
        low=1.0,
        max_samples=100_000,
        threshold=0.95,
        random_state=None
    ):
        self.high = high
        self.low = low
        self.max_samples = max_samples
        self.threshold = threshold
        self.random_state = random_state

    @timeit
    def fit(self, X):
        random_state = check_random_state(self.random_state)
        frequency = X.nunique()

        logger.info(frequency)

        self.categorical_features_ = \
            [c for c in X if c.startswith(CATEGORICAL_PREFIX)]
        self.multi_value_categorical_features_ = \
            [c for c in X if c.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]
        self.numerical_features_ = \
            [c for c in X if c.startswith(NUMERICAL_PREFIX)]
        self.time_features_ = [c for c in X if c.startswith(TIME_PREFIX)]

        self.n_samples_, _ = X.shape
        self.dropped_features_ = X.columns[(frequency == 1)]

        if len(self.categorical_features_) > 0:
            self.dropped_features_ = self.dropped_features_.union(
                list(
                    itertools.compress(
                        self.categorical_features_,
                        frequency[self.categorical_features_]
                        == self.n_samples_
                    )
                )
            )

        if len(self.numerical_features_) > 0:
            self.data_min_, self.data_max_ = np.nanpercentile(
                X[self.numerical_features_],
                [self.low, self.high],
                axis=0
            )

            if self.max_samples < self.n_samples_:
                indices = random_state.choice(
                    self.n_samples_,
                    self.max_samples,
                    replace=False
                )

                corr = X[self.numerical_features_].iloc[indices].corr()

            else:
                corr = X[self.numerical_features_].corr()

            triu = np.triu(corr, k=1)
            triu = np.abs(triu)
            triu = np.nan_to_num(triu)

            self.dropped_features_ = self.dropped_features_.union(
                list(
                    itertools.compress(
                        self.numerical_features_,
                        np.any(triu > self.threshold, axis=0)
                    )
                )
            )

        if len(self.time_features_) > 0:
            self.dropped_features_ = self.dropped_features_.union(
                self.time_features_
            )

        logger.info(f'dropped_features={self.dropped_features_}')

        return self

    @timeit
    def transform(self, X):
        logger.info(f'before engineering: X.shape={X.shape}')

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

        X = X.drop(columns=self.dropped_features_)

        logger.info(f'after engineering: X.shape={X.shape}')

        return X


class AutoSSLClassifier(BaseEstimator):
    @property
    def predict_proba(self):
        return self.model_.predict_proba

    def __init__(
        self,
        class_weight=None,
        cv=5,
        max_samples=100_000,
        n_jobs=1,
        n_trials=25,
        proportion=0.1,
        random_state=None,
        timeout=None
    ):
        self.class_weight = class_weight
        self.cv = cv
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.proportion = proportion
        self.random_state = random_state
        self.timeout = timeout

    def fit(self, X, y, **fit_params):
        timer = Timer(time_budget=self.timeout)

        timer.start()

        n_samples, _ = X.shape
        n_neg_samples = np.sum(y == -1)
        n_pos_samples = np.sum(y == 1)
        low = 50.0 * self.proportion * n_neg_samples \
            / (n_neg_samples + n_pos_samples)
        high = 100.0 - low
        is_labeled = y != 0
        X_labeled = X[is_labeled]
        y_labeled = y[is_labeled]
        iter_time = 0.0

        self.model_ = AutoNoisyClassifier(
            class_weight=self.class_weight,
            cv=self.cv,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            n_trials=100,
            random_state=self.random_state
        )

        while timer.get_remaining_time() - iter_time > 0:
            start_time = time.perf_counter()

            self.model_.fit(
                X_labeled,
                y_labeled,
                timeout=timer.get_remaining_time(),
                **fit_params
            )

            if timer.get_remaining_time() <= 0:
                break

            if is_labeled.sum() == n_samples:
                break

            y_score = np.full(n_samples, np.nan)
            y_score[~is_labeled] = self.model_.predict_proba(X[~is_labeled])
            low_value, high_value = np.nanpercentile(y_score, [low, high])
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


class AutoPUClassifier(BaseEstimator):
    def __init__(
        self,
        class_weight=None,
        cv=5,
        max_samples=100_000,
        n_jobs=1,
        n_trials=25,
        random_state=None,
        timeout=None
    ):
        self.class_weight = class_weight
        self.cv = cv
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.timeout = timeout

    def fit(self, X, y, **fit_params):
        timer = Timer(time_budget=self.timeout)

        timer.start()

        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape
        n_pos_samples = np.sum(y == 1)
        sample_indices = np.arange(n_samples)
        sample_indices_unlabeled = sample_indices[y == 0]
        sample_indices_positive = sample_indices[y == 1]
        iter_time = 0.0

        model = AutoNoisyClassifier(
            class_weight=self.class_weight,
            cv=self.cv,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            n_trials=100,
            random_state=self.random_state
        )

        self.models_ = []

        while timer.get_remaining_time() - iter_time > 0:
            start_time = time.perf_counter()

            sample_indices_sampled = np.union1d(
                random_state.choice(
                    sample_indices_unlabeled,
                    n_pos_samples,
                    replace=False
                ),
                sample_indices_positive
            )

            m = clone(model)

            m.fit(
                X.iloc[sample_indices_sampled],
                y.iloc[sample_indices_sampled],
                timeout=timer.get_remaining_time(),
                **fit_params
            )

            self.models_.append(m)

            iter_time = time.perf_counter() - start_time

        return self

    def predict_proba(self, X):
        for i, model in enumerate(self.models_):
            p = model.predict_proba(X)

            if i == 0:
                probas = p
            else:
                probas += p

        return probas / len(self.models_)


class AutoNoisyClassifier(BaseEstimator):
    def __init__(
        self,
        class_weight=None,
        cv=5,
        max_samples=100_000,
        n_jobs=1,
        n_trials=25,
        random_state=None,
        timeout=None
    ):
        self.class_weight = class_weight
        self.cv = cv
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.random_state = random_state
        self.timeout = timeout

    def fit(self, X, y, timeout=None, **fit_params):
        timer = Timer(time_budget=self.timeout)

        timer.start()

        cv = check_cv(self.cv, y=y, classifier=True)
        random_state = check_random_state(self.random_state)
        n_samples, _ = X.shape

        if n_samples > self.max_samples:
            if isinstance(cv, TimeSeriesSplit):
                X = X.iloc[-self.max_samples:]
                y = y.iloc[-self.max_samples:]
            else:
                sample_indices = random_state.choice(
                    n_samples,
                    self.max_samples,
                    replace=False
                )

                sample_indices.sort()

                X = X.iloc[sample_indices]
                y = y.iloc[sample_indices]

        if timeout is None:
            if self.timeout is None:
                timeout = None
            else:
                timeout = timer.get_remaining_time()

        self.model_ = OGBMClassifier(
            class_weight=self.class_weight,
            cv=cv,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            random_state=random_state,
            timeout=timeout
        )

        self.model_.fit(X, y, **fit_params)

        return self

    def predict_proba(self, X):
        probas = self.model_.predict_proba(X)

        return probas[:, 1]


class Model(object):
    def __init__(
        self,
        info: dict,
        class_weight='balanced',
        cv=4,
        high=99.9,
        low=0.1,
        max_samples=30_000,
        n_jobs=-1,
        n_trials=None,
        random_state=0
    ):
        self.class_weight = class_weight
        self.cv = cv
        self.high = high
        self.info = info
        self.low = low
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

        self.engineer_ = Enginner(
            high=self.high,
            max_samples=self.max_samples,
            low=self.low,
            random_state=self.random_state
        )

        self.engineer_.fit(X)

        X = self.engineer_.transform(X)

        if self.info['task'] == 'ssl':
            klass = AutoSSLClassifier
        elif self.info['task'] == 'pu':
            klass = AutoPUClassifier
        elif self.info['task'] == 'noisy':
            klass = AutoNoisyClassifier

        timeout = \
            0.75 * self.info['time_budget'] - self._timer.get_elapsed_time()

        self.model_ = klass(
            class_weight=self.class_weight,
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

        probas = self.model_.predict_proba(X)

        return pd.Series(probas)

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
