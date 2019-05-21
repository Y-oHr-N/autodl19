import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn==0.21rc2')

import pandas as pd

from automllib.automl import AutoMLClassifier


class Model(object):
    def __init__(self, info):
        self.info = info

    def fit(self, Xs, y, timeout):
        self.model_ = AutoMLClassifier(self.info)

        self.model_.fit(Xs, y, timeout)

    def predict(self, X, timeout):
        probas = self.model_.predict_proba(X, timeout)

        return pd.Series(probas[:, 1])
