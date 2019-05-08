import logging

from typing import Type
from typing import Union

import pandas as pd

from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils import check_array

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit

logger = logging.getLogger(__name__)


class TimeVectorizer(BaseEstimator, TransformerMixin):
    _attributes = [
        # 'year',
        'weekofyear',
        'dayofyear',
        'quarter',
        'month',
        'day',
        'weekday',
        'hour',
        'minute',
        'second'
    ]

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'TimeVectorizer':
        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        dfs = []

        for column in X:
            df = pd.DataFrame()

            for attr in self._attributes:
                df[f'{column}_{attr}'] = getattr(X[column].dt, attr)

            dfs.append(df)

        Xt = pd.concat(dfs, axis=1)
        _, n_features = Xt.shape

        logger.info(
            f'{self.__class__.__name__} extracts {n_features} features.'
        )

        return Xt


class MultiValueCategoricalVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        dtype: Union[str, Type] = 'float64',
        lowercase: bool = True,
        n_features_per_column: int = 1048576
    ):
        self.dtype = dtype
        self.lowercase = lowercase
        self.n_features_per_column = n_features_per_column

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'MultiValueCategoricalVectorizer':
        X = check_array(X, dtype=None, estimator=self)
        v = HashingVectorizer(
            dtype=self.dtype,
            lowercase=self.lowercase,
            n_features=self.n_features_per_column
        )

        self.vectorizers_ = [clone(v).fit(column) for column in X.T]

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = check_array(X, dtype=None, estimator=self)
        Xs = [
            self.vectorizers_[j].transform(
                column
            ) for j, column in enumerate(X.T)
        ]
        Xt = hstack(Xs)
        _, n_features = Xt.shape

        logger.info(
            f'{self.__class__.__name__} extracts {n_features} features.'
        )

        return Xt
