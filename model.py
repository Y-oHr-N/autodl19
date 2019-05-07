import copy
import logging
import os

from typing import Any
from typing import Dict

os.system("pip3 install imbalanced-learn")
os.system("pip3 install lightgbm")
os.system("pip3 install optuna")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install scikit-learn==0.21rc2")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from automllib.compose import make_mixed_transformer
from automllib.compose import make_model
from automllib.constants import MAIN_TABLE_NAME
from automllib.constants import ONE_DIM_ARRAY_TYPE
from automllib.constants import TWO_DIM_ARRAY_TYPE
from automllib.table_join import Config
from automllib.table_join import merge_table
from automllib.utils import timeit

np.random.seed(0)

logger = logging.getLogger(__name__)


class Model(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, info: Dict[str, Any]) -> None:
        self.info = info

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(
            self,
            ['config_', 'estimator_' 'preprocessor_' 'tables_']
        )

    @timeit
    def fit(
        self,
        Xs: Dict[str, TWO_DIM_ARRAY_TYPE],
        y: ONE_DIM_ARRAY_TYPE,
        timeout: float = None
    ) -> 'Model':
        self.config_ = Config(self.info)
        self.tables_ = copy.deepcopy(Xs)

        X = merge_table(Xs, self.config_)

        X.sort_values(self.info['time_col'], inplace=True)

        self.preprocessor_ = make_mixed_transformer()

        X = self.preprocessor_.fit_transform(X)

        self.estimator_ = make_model()

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            shuffle=False
        )

        fit_params = {
            'optunasearchcv__early_stopping_rounds': 10,
            'optunasearchcv__eval_set': [(X_valid, y_valid)],
            'optunasearchcv__verbose': False
        }

        self.estimator_.fit(X_train, y_train, **fit_params)

        try:
            best_score = self.estimator_._final_estimator.best_score_

            logger.info(f'The best score is {best_score:.3f}')

        except Exception as e:
            logger.exception(e)

        return self

    @timeit
    def predict(
        self,
        X_test: TWO_DIM_ARRAY_TYPE,
        timeout: float = None
    ) -> ONE_DIM_ARRAY_TYPE:
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
