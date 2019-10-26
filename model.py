import logging
import os
import pickle
import time

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q optgbm')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn>=0.21.0')

import colorlog
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMClassifier

from automllib.utils import Timeit

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s'
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

timeit = Timeit(logger=logger)


NUMERICAL_PREFIX = "n_"
CATEGORY_PREFIX = "c_"
TIME_PREFIX = "t_"
MULTI_VALUE_CATEGORICAL_PREFIX = "multi-cat"


@timeit
def feature_engineer(df):
    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)

    for c in [c for c in df if c.startswith(CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: hash(x))


class Feature_enginner():
    def __init__(self):
        self.numerical_percentile = {}
        self.time_extraction = {}

    def fit(self, X):
        secondsinminute = 60.0
        secondsinhour = 60.0 * secondsinminute
        secondsinday = 24.0 * secondsinhour
        secondsinweekday = 7.0 * secondsinday
        secondsinyear = 365.0 * secondsinday
        secondsinmonth = secondsinyear / 12.0
        for c in [c for c in X if c.startswith(TIME_PREFIX)]:
            time_properties = []
            duration = (X[c].max() - X[c].min()).total_seconds()
            if duration > 2*secondsinminute:
                time_properties.append('second')
            if duration > 2*secondsinhour \
                    and len(pd.unique(X[c].dt.minute)) > 2:
                time_properties.append('minute')
            if duration > 2*secondsinday \
                    and len(pd.unique(X[c].dt.hour)) > 2:
                time_properties.append('hour')
            if duration > 2*secondsinweekday \
                    and len(pd.unique(X[c].dt.weekday)) > 2:
                time_properties.append('weekday')
            if duration > 2*secondsinmonth \
                    and len(pd.unique(X[c].dt.day)) > 2:
                time_properties.append('day')
            if duration > 2*secondsinyear:
                if len(pd.unique(X[c].dt.month)) > 2:
                    time_properties.append('month')
                if len(pd.unique(X[c].dt.quarter)) > 2:
                    time_properties.append('quarter')
            self.time_extraction[c] = time_properties
        for c in [c for c in X if c.startswith(NUMERICAL_PREFIX)]:
            data_min, data_max = np.nanpercentile(
                X[c],
                [1.0, 99.0],
                axis=0
            )
            self.numerical_percentile[c] = [data_min, data_max]

    def transform(self, X):
        for c in [c for c in X if c.startswith(TIME_PREFIX)]:
            for time_properties in self.time_extraction[c]:
                X[c + "_" + time_properties] = getattr(X[c].dt, time_properties)
            X.drop(c, axis=1, inplace=True)
        for c in [c for c in X if c.startswith(CATEGORY_PREFIX)]:
            X[c] = X[c].apply(lambda x: hash(x))
        for c in [c for c in X if c.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]:
            X[c] = X[c].apply(lambda x: hash(x))
        for c in [c for c in X if c.startswith(NUMERICAL_PREFIX)]:
            X[c] = np.clip(X[c], self.numerical_percentile[c][0], self.numerical_percentile[c][1])
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

@timeit
def sample(X, y, nrows):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample


class AutoSSLClassifier:
    def __init__(self):
        self.iter = 5
        self.label_data = 500
        self.model = None

    def fit(self, X, y):
        X_label, y_label, X_unlabeled, y_unlabeled = self._split_by_label(X, y)
        y_n_cnt, y_p_cnt = y_label.value_counts()

        y_n = max(int(self.label_data * (y_n_cnt * 1.0 / len(y_label))), 1)
        y_p = max(int(self.label_data * (y_p_cnt * 1.0 / len(y_label))), 1)

        for _ in range(self.iter):
            if X_unlabeled.shape[0] < self.label_data:
                break

            self.model = AutoNoisyClassifier()

            self.model.fit(X_label, y_label)

            y_hat = self.model.predict(X_unlabeled)

            if len(set(y_hat)) == 1:
                break

            idx = np.argsort(y_hat)
            y_p_idx = idx[-y_p:]
            y_n_idx = idx[:y_n]
            X_label = pd.concat((X_label, X_unlabeled.iloc[list(y_p_idx) + list(y_n_idx), :]))
            y_label = pd.concat((y_label, pd.Series([1]*len(y_p_idx) + [-1]*len(y_n_idx))))
            y_label.index = X_label.index
            X_unlabeled = X_unlabeled.iloc[idx[y_n:-y_p], :]

        return self

    def predict(self, X):
        return self.model.predict(X)

    def _split_by_label(self, X, y):
        y_label = pd.concat([y[y == -1], y[y == 1]])
        X_label = X.loc[y_label.index, :]
        y_unlabeled = y[y == 0]
        X_unlabeled = X.loc[y_unlabeled.index, :]

        return X_label, y_label, X_unlabeled, y_unlabeled


class AutoPUClassifier:
    def __init__(self):
        self.iter = 10
        self.models = []

    def fit(self, X, y):
        for _ in range(self.iter):
            X_sample, y_sample = self._negative_sample(X, y)

            model = AutoNoisyClassifier()

            model.fit(X_sample, y_sample)

            self.models.append(model)

        return self

    def predict(self, X):
        for idx, model in enumerate(self.models):
            p = model.predict(X)

            if idx == 0:
                prediction = p
            else:
                prediction += p

        return prediction / len(self.models)

    def _negative_sample(self, X, y):
        y_n_cnt, y_p_cnt = y.value_counts()
        y_n_sample = y_p_cnt if y_n_cnt > y_p_cnt else y_n_cnt
        y_sample = pd.concat([y[y == 0].sample(y_n_sample), y[y == 1]])
        x_sample = X.loc[y_sample.index, :]

        return x_sample, y_sample


class AutoNoisyClassifier:
    def fit(self, X, y):
        X_sample, y_sample = sample(X, y, 30_000)

        self.model = OGBMClassifier(cv=3, n_jobs=4, random_state=0)

        self.model.fit(X_sample, y_sample, eval_metric='auc')

        return self

    def predict(self, X):
        return pd.Series(self.model.predict_proba(X)[:, 1])


class Model:
    def __init__(self, info: dict):
        self.info = info
        self.transformer = None

    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.transformer = Feature_enginner()
        X = self.transformer.fit_transform(X)
        if self.info['task'] == 'ssl':
            self.model = AutoSSLClassifier()
        elif self.info['task'] == 'pu':
            self.model = AutoPUClassifier()
        elif self.info['task'] == 'noisy':
            self.model = AutoNoisyClassifier()

        self.model.fit(X, y)

    @timeit
    def predict(self, X: pd.DataFrame):
        X = self.transformer.transform(X)

        return self.model.predict(X)

    @timeit
    def save(self, directory: str):
        pickle.dump(
            self.model,
            open(os.path.join(directory, 'model.pkl'), 'wb')
        )

    @timeit
    def load(self, directory: str):
        self.model = pickle.load(
            open(os.path.join(directory, 'model.pkl'), 'rb')
        )
