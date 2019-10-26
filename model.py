import logging
import os
import pickle
import time

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q git+https://github.com/Y-oHr-N/OptGBM.git')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn>=0.21.0')

import colorlog
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMClassifier

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


NUMERICAL_PREFIX = "n_"
CATEGORY_PREFIX = "c_"
TIME_PREFIX = "t_"


@timeit
def feature_engineer(df):
    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)

    for c in [c for c in df if c.startswith(CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: hash(x))


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
    def __init__(self, tuning_time=None):
        self.iter = 5
        self.label_data = 500
        self.model = None
        self.tuning_time = tuning_time

    def fit(self, X, y):
        X_label, y_label, X_unlabeled, y_unlabeled = self._split_by_label(X, y)
        y_n_cnt, y_p_cnt = y_label.value_counts()

        y_n = max(int(self.label_data * (y_n_cnt * 1.0 / len(y_label))), 1)
        y_p = max(int(self.label_data * (y_p_cnt * 1.0 / len(y_label))), 1)

        for _ in range(self.iter):
            if X_unlabeled.shape[0] < self.label_data:
                break

            self.model = AutoNoisyClassifier(tuning_time=self.tuning_time/self.iter)

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
    def __init__(self, tuning_time=None):
        self.iter = 10
        self.models = []
        self.tuning_time = tuning_time

    def fit(self, X, y):
        for _ in range(self.iter):
            X_sample, y_sample = self._negative_sample(X, y)

            model = AutoNoisyClassifier(tuning_time=self.tuning_time/self.iter)

            model.fit(X_sample, y_sample)

            self.models.append(model)

        return self

    def predict(self, X):
        for idx, model in enumerate(self.models):
            p = model.set_params(n_jobs=1).predict(X)

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
    def __init__(self, max_samples=100_000, tuning_time=None):
        self.max_smaples = max_samples
        self.tuning_time = tuning_time

    def fit(self, X, y):
        X_sample, y_sample = sample(X, y, self.max_samples)

        self.model = OGBMClassifier(
            cv=3,
            n_jobs=4,
            n_trials=None,
            random_state=0,
            timeout=self.tuning_time
        )

        self.model.fit(X_sample, y_sample, eval_metric='auc')

        return self

    def predict(self, X):
        return pd.Series(self.model.predict_proba(X)[:, 1])


class Model:
    def __init__(self, info: dict):
        self.info = info
        self.timer = Timer(self.info['time_budget'])
        self.timer.start()

    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        feature_engineer(X)

        logger.info(f'X.shape = {X.shape}')

        tuning_time = 0.8 * self.info["time_budget"] - self.timer.get_elapsed_time()
        if self.info['task'] == 'ssl':
            self.model = AutoSSLClassifier(tuning_time=tuning_time)
        elif self.info['task'] == 'pu':
            self.model = AutoPUClassifier(tuning_time=tuning_time)
        elif self.info['task'] == 'noisy':
            self.model = AutoNoisyClassifier(tuning_time=tuning_time)

        self.model.fit(X, y)

    @timeit
    def predict(self, X: pd.DataFrame):
        feature_engineer(X)

        logger.info(f'X.shape = {X.shape}')

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
