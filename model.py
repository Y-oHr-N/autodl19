import os

os.system('pip3 install hyperopt numpy lightgbm pandas>=0.24.2 scikit-learn')

import copy
import pandas as pd

from package.automl import predict
from package.automl import train
from package.constants import MAIN_TABLE_NAME
from package.merge import merge_table
from package.preprocess import clean_df
from package.preprocess import clean_tables
from package.preprocess import feature_engineer
from package.utils import Config
from package.utils import timeit


class Model(object):
    def __init__(self, info):
        self.config = Config(info)
        self.tables = None

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)

        clean_tables(Xs)

        X = merge_table(Xs, self.config)

        clean_df(X)
        feature_engineer(X, self.config)
        train(X, y, self.config)

    @timeit
    def predict(self, X_test, time_remain):

        Xs = self.tables
        main_table = Xs[MAIN_TABLE_NAME]
        main_table = pd.concat([main_table, X_test], keys=['train', 'test'])
        main_table.index = main_table.index.map(lambda x: f'{x[0]}_{x[1]}')
        Xs[MAIN_TABLE_NAME] = main_table

        clean_tables(Xs)

        X = merge_table(Xs, self.config)

        clean_df(X)
        feature_engineer(X, self.config)

        X = X[X.index.str.startswith('test')]
        X.index = X.index.map(lambda x: int(x.split('_')[1]))

        X.sort_index(inplace=True)

        result = predict(X, self.config)

        return pd.Series(result)
