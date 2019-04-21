import os
from typing import Any
from typing import Dict

os.system("pip3 install imbalanced-learn")
os.system("pip3 install lightgbm")
os.system("pip3 install optuna")
os.system("pip3 install pandas==0.24.2")

import copy
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin

from automllib.compose import make_mixed_transformer
from automllib.constants import MAIN_TABLE_NAME
from automllib.merge import Config
from automllib.merge import merge_table
from automllib.train import train
from automllib.utils import timeit

np.random.seed(0)


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

        X = merge_table(Xs, self.config_)

        self.preprocessor_ = make_mixed_transformer()

        X = self.preprocessor_.fit_transform(X)

        self.estimator_ = train(X, y)

        return self

    @timeit
    def predict(self, X_test: pd.DataFrame, time_remain: float) -> pd.Series:
        Xs = self.tables_
        main_table = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f'{x[0]}_{x[1]}')
        Xs[MAIN_TABLE_NAME] = main_table

        X = merge_table(Xs, self.config_)

        X = X[X.index.str.startswith('test')]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))

        X.sort_index(inplace=True)

        X = self.preprocessor_.transform(X)

        result = self.estimator_.predict_proba(X)

        return pd.Series(result[:, 1])
