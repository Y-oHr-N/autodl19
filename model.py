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
import lightgbm as lgb
import numpy as np
import pandas as pd

from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
from sklearn.model_selection import train_test_split

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


def _hyperopt(X, y, params):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.8, 1.0, 0.1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }

    def objective(hyperparams):
        model = lgb.train(
            {**params, **hyperparams},
            train_data,
            300,
            valid_data,
            early_stopping_rounds=30,
            verbose_eval=0
        )

        score = model.best_score["valid_0"][params["metric"]]

        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, trials=trials,
                algo=tpe.suggest, max_evals=2, verbose=1,
                rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)

    logger.info(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

    return hyperparams


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
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 1,
            "num_threads": 4
        }

        X_sample, y_sample = sample(X, y, 30000)
        hyperparams = _hyperopt(X_sample, y_sample, params)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=1
        )
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train(
            {**params, **hyperparams},
            train_data,
            500,
            valid_data,
            early_stopping_rounds=30,
            verbose_eval=100
        )

        self.model.free_dataset()

        return self

    def predict(self, X):
        return self.model.predict(X)


NUMERICAL_PREFIX = "n_"
CATEGORY_PREFIX = "c_"
TIME_PREFIX = "t_"


@timeit
def feature_engineer(df):
    transform_categorical_hash(df)
    transform_datetime(df)


@timeit
def transform_datetime(df):
    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
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


class Model:
    def __init__(self, info: dict):
        logger.info(f"Info:\n {info}")

        self.model = None
        self.task = info['task']
        self.train_time_budget = info['time_budget']
        # self.pred_time_budget = info.get('pred_time_budget')
        self.cols_dtype = info['schema']

        self.dtype_cols = {'cat': [], 'num': [], 'time': []}

        for key, value in self.cols_dtype.items():
            if value == 'cat':
                self.dtype_cols['cat'].append(key)
            elif value == 'num':
                self.dtype_cols['num'].append(key)
            elif value == 'time':
                self.dtype_cols['time'].append(key)

    @timeit
    def train(self, X: pd.DataFrame, y: pd.Series):
        start_time = time.time()

        feature_engineer(X)

        logger.info(f"Remain time: {self.train_time_budget - (time.time() - start_time)}")

        if self.task == 'ssl':
            self.model = AutoSSLClassifier()
        elif self.task == 'pu':
            self.model = AutoPUClassifier()
        elif self.task == 'noisy':
            self.model = AutoNoisyClassifier()

        self.model.fit(X, y)

    @timeit
    def predict(self, X: pd.DataFrame):
        # start_time = time.time()

        feature_engineer(X)

        # logger.info(f"Remain time: {self.pred_time_budget - (time.time() - start_time)}")

        prediction = self.model.predict(X)

        return pd.Series(prediction)

    @timeit
    def save(self, directory: str):
        pickle.dump(
            self.model, open(os.path.join(directory, 'model.pkl'), 'wb'))

    @timeit
    def load(self, directory: str):
        self.model = pickle.load(
            open(os.path.join(directory, 'model.pkl'), 'rb'))
