import copy
import logging
import os

from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

os.system("pip3 install colorlog")
os.system("pip3 install imbalanced-learn")
os.system("pip3 install lightgbm")
os.system("pip3 install optuna")
os.system("pip3 install pandas==0.24.2")
os.system("pip3 install scikit-learn==0.21rc2")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from automllib.compose import Maker
from automllib.constants import MAIN_TABLE_NAME
from automllib.constants import ONE_DIM_ARRAY_TYPE
from automllib.constants import TWO_DIM_ARRAY_TYPE
from automllib.table_join import Config
from automllib.table_join import merge_table
from automllib.utils import Timeit

np.random.seed(0)

logger = logging.getLogger(__name__)


class Model(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        info: Dict[str, Any],
        cv: int = 3,
        estimator_type: str = 'classifier',
        lowercase: bool = False,
        max_iter: int = 10,
        metric: str = 'auc',
        n_features_per_column: int = 32,
        n_jobs: int = -1,
        random_state: Union[int, np.random.RandomState] = 0,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        scoring: Union[str, Callable[..., float]] = 'roc_auc',
        shuffle: bool = False,
        subsample: Union[int, float] = 100_000,
        verbose: int = 1
    ) -> None:
        self.cv = cv
        self.estimator_type = estimator_type
        self.info = info
        self.lowercase = lowercase
        self.max_iter = max_iter
        self.metric = metric
        self.n_features_per_column = n_features_per_column
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.scoring = scoring
        self.shuffle = shuffle
        self.subsample = subsample
        self.verbose = verbose

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(
            self,
            ['config_', 'estimator_' 'preprocessor_' 'tables_']
        )

    @Timeit(logger)
    def fit(
        self,
        Xs: Dict[str, TWO_DIM_ARRAY_TYPE],
        y: ONE_DIM_ARRAY_TYPE,
        timeout: float = None
    ) -> 'Model':
        self.config_ = Config(self.info)
        self.tables_ = copy.deepcopy(Xs)
        self.maker_ = Maker(
            self.estimator_type,
            cv=TimeSeriesSplit(self.cv),
            lowercase=self.lowercase,
            max_iter=self.max_iter,
            metric=self.metric,
            n_features_per_column=self.n_features_per_column,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            scoring=self.scoring,
            shuffle=self.shuffle,
            subsample=self.subsample,
            verbose=self.verbose
        )
        self.model_ = self.maker_.make_model()

        X = merge_table(Xs, self.config_)

        self.model_.fit(X, y)

        return self

    @Timeit(logger)
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

        result = self.model_.predict_proba(X)

        return pd.Series(result[:, 1])
