import copy
import logging
import os

from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

os.system("pip3 install -q colorlog")
os.system("pip3 install -q imbalanced-learn")
os.system("pip3 install -q lightgbm")
os.system("pip3 install -q optuna")
os.system("pip3 install -q pandas==0.24.2")
os.system("pip3 install -q scikit-learn==0.21rc2")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target

from automllib.compose import PipelineMaker
from automllib.constants import MAIN_TABLE_NAME
from automllib.constants import ONE_DIM_ARRAY_TYPE
from automllib.constants import TWO_DIM_ARRAY_TYPE
from automllib.table_join import Config
from automllib.table_join import merge_table
from automllib.utils import Timeit

logger = logging.getLogger(__name__)


class Model(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        info: Dict[str, Any],
        cv: int = 3,
        early_stopping_rounds: int = 10,
        lowercase: bool = False,
        max_iter: int = 10,
        metric: str = 'auc',
        n_estimators: int = 100,
        n_features_per_column: int = 32,
        n_jobs: int = -1,
        random_state: Union[int, np.random.RandomState] = 0,
        sampling_strategy: Union[str, float, Dict[str, int]] = 'auto',
        scoring: Union[str, Callable[..., float]] = 'roc_auc',
        shuffle: bool = False,
        subsample: Union[int, float] = 100_000,
        validation_size: float = 0.1,
        verbose: int = 1
    ) -> None:
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.info = info
        self.lowercase = lowercase
        self.max_iter = max_iter
        self.metric = metric
        self.n_estimators = n_estimators
        self.n_features_per_column = n_features_per_column
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.scoring = scoring
        self.shuffle = shuffle
        self.subsample = subsample
        self.validation_size = validation_size
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
        self.maker_ = PipelineMaker(
            type_of_target(y),
            cv=TimeSeriesSplit(self.cv),
            lowercase=self.lowercase,
            max_iter=self.max_iter,
            metric=self.metric,
            n_estimators=self.n_estimators,
            n_features_per_column=self.n_features_per_column,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            sampling_strategy=self.sampling_strategy,
            scoring=self.scoring,
            shuffle=self.shuffle,
            subsample=self.subsample,
            verbose=self.verbose
        )
        self.transformer_ = self.maker_.make_transformer()
        self.sampler_ = self.maker_.make_sampler()
        self.search_cv_ = self.maker_.make_search_cv()

        X = merge_table(Xs, self.config_)

        X = X.sort_values(self.info['time_col'])
        y = y.loc[X.index]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            random_state=self.random_state,
            shuffle=self.shuffle,
            test_size=self.validation_size
        )
        X_train = self.transformer_.fit_transform(X_train)
        X_valid = self.transformer_.transform(X_valid)
        X_train, y_train = self.sampler_.fit_resample(X_train, y_train)
        fit_params = {
            'lgbmclassifier__early_stopping_rounds': self.early_stopping_rounds,
            'lgbmclassifier__eval_set': [(X_valid, y_valid)],
            'lgbmclassifier__verbose': False
        }

        self.search_cv_.fit(X, y, **fit_params)

        return self

    @Timeit(logger)
    def predict(
        self,
        X_test: TWO_DIM_ARRAY_TYPE,
        timeout: float = None
    ) -> ONE_DIM_ARRAY_TYPE:
        Xs = self.tables_
        Xs[MAIN_TABLE_NAME] = X_test

        X = merge_table(Xs, self.config_)

        X = X.sort_index()

        X = self.transformer_.transform(X)
        result = self.search_cv_.predict_proba(X)

        return pd.Series(result[:, 1])
