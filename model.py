import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn>=0.21.0')

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

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
        related_tables = Xs.copy()
        X = related_tables.pop('main')

        if isinstance(self.info, dict) and 'time_col' in self.info:
            X = X.sort_values(self.info['time_col'], na_position='first')
            y = y.loc[X.index]

        self.model_ = AutoMLClassifier(
            cv=TimeSeriesSplit(5),
            info=self.info,
            related_tables=related_tables,
            shuffle=False
        )

        self.model_.fit(X, y)

    def predict(self, X, timeout):
        probas = self.model_.predict_proba(X)

        return pd.Series(probas[:, 1])
