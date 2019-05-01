import logging

import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer

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
    def transform(
        self,
        X: TWO_DIM_ARRAY_TYPE
    ) -> ONE_DIM_ARRAY_TYPE:
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
    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'MultiValueCategoricalVectorizer':
        self.vectorizers_ = []

        for column in X.T:
            vectorizer = HashingVectorizer()

            vectorizer.fit(column)

            self.vectorizers_.append(vectorizer)

        return self

    @timeit
    def transform(
        self,
        X: TWO_DIM_ARRAY_TYPE
    ) -> ONE_DIM_ARRAY_TYPE:
        count_matrix = []

        for column, vectorizer in zip(X.T, self.vectorizers_):
            count_matrix.append(vectorizer.transform(column))

        Xt = hstack(tuple(count_matrix))
        _, n_features = Xt.shape

        logger.info(
            f'{self.__class__.__name__} extracts {n_features} features.'
        )

        return Xt
