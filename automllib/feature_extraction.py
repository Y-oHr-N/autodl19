import logging

from typing import Any
from typing import Type
from typing import Union

import pandas as pd

from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils.validation import check_is_fitted

from .base import BaseTransformer
from .base import ONE_DIM_ARRAY_TYPE
from .base import TWO_DIM_ARRAY_TYPE
from .utils import timeit

logger = logging.getLogger(__name__)


class TimeVectorizer(BaseTransformer):
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

    def __init__(self, **params: Any) -> None:
        pass

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        pass

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'TimeVectorizer':
        self._check_params()

        X = self._check_array(X)

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

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


class MultiValueCategoricalVectorizer(BaseTransformer):
    def __init__(
        self,
        dtype: Union[str, Type] = 'float64',
        lowercase: bool = True,
        n_features_per_column: int = 1048576
    ) -> None:
        self.dtype = dtype
        self.lowercase = lowercase
        self.n_features_per_column = n_features_per_column

    def _check_params(self) -> None:
        pass

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, ['vectorizers_'])

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'MultiValueCategoricalVectorizer':
        self._check_params()

        X = self._check_array(X)
        v = HashingVectorizer(
            dtype=self.dtype,
            lowercase=self.lowercase,
            n_features=self.n_features_per_column
        )

        self.vectorizers_ = [clone(v).fit(column) for column in X.T]

        return self

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

        X = self._check_array(X)
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
