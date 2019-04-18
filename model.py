import os
from typing import Any
from typing import Dict

os.system("pip3 install imbalanced-learn")
os.system("pip3 install lightgbm")
os.system("pip3 install optuna")
os.system("pip3 install pandas==0.24.2")

import copy
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin

from automllib.constants import MAIN_TRAIN_TABLE_NAME
from automllib.constants import MAIN_TEST_TABLE_NAME
from automllib.merge import Config
from automllib.merge import merge_table
from automllib.merge import merge_table_test
from automllib.preprocessing import clean_df
from automllib.preprocessing import clean_tables
from automllib.preprocessing import feature_engineer
from automllib.preprocessing import delete_columns
from automllib.train import resample
from automllib.train import train
from automllib.utils import timeit


class Model(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, info: Dict[str, Any]) -> None:
        self.info = info

    @timeit
    def fit(
        self,
        Xs: Dict[str, pd.DataFrame],
        y: pd.Series,
        time_ramain: float
    ) -> 'Model':
        self.config_ = Config(self.info)
        self.tables_ = copy.deepcopy(Xs)

        clean_tables(Xs)

        feature_engineer(Xs, self.config_)
        X = merge_table(Xs, self.config_)
        clean_df(X)
        delete_columns(X)
        X, y = resample(X, y, random_state=0)
        self.estimator_ = train(X, y, n_jobs=-1, n_trials=16, random_state=0)

        return self

    @timeit
    def predict(self, X_test: pd.DataFrame, time_remain: float) -> pd.Series:
        Xs = {}
        Xs[MAIN_TEST_TABLE_NAME] = X_test
        clean_tables(Xs)

        feature_engineer(Xs, self.config_)
        X = merge_table_test(Xs[MAIN_TEST_TABLE_NAME], self.config_)

        clean_df(X)
        delete_columns(X)
        result = self.estimator_.predict_proba(X)

        return pd.Series(result[:, 1])
