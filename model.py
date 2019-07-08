import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn>=0.21.0')

import pandas as pd

from automllib.automl import AutoMLClassifier


class Model(object):
    @property
    def best_params_(self):
        return self.model_.best_params_

    @property
    def best_score_(self):
        return self.model_.best_score_

    def __init__(self, info):
        self.info = info

    def fit(self, Xs, y, timeout):
        params = self.info.copy()
        params['n_jobs'] = -1
        params['random_state'] = 0
        params['verbose'] = 1
        related_tables = Xs.copy()
        X = related_tables.pop('main')

        self.model_ = AutoMLClassifier(**params)

        self.model_.fit(X, y, related_tables=related_tables)

    def predict(self, X, timeout):
        probas = self.model_.predict_proba(X)

        return pd.Series(probas[:, 1])
