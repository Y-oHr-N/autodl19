import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn>=0.21.0')

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from automllib.automl import AutoMLClassifier
from automllib.constants import MAIN_TABLE_NAME


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
        X = related_tables.pop(MAIN_TABLE_NAME)

        self.model_ = AutoMLClassifier(
            self.info,
            related_tables,
            cv=TimeSeriesSplit(3),
            lowercase=False,
            max_depth=7,
            n_estimators=300,
            n_jobs=-1,
            random_state=0,
            shuffle=False,
            validation_fraction=0.2,
            verbose=1
        )

        self.model_.fit(X, y, timeout=timeout)

    def predict(self, X, timeout):
        probas = self.model_.predict_proba(X)

        return pd.Series(probas[:, 1])
