import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .constants import MULTI_VALUE_CATEGORICAL_PREFIX
from .utils import timeit
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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> 'TimeVectorizer':
        return self

    def transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        dfs = []

        for column in X:
            df = pd.DataFrame()

            for attr in self._attributes:
                df[f'{column}_{attr}'] = getattr(X[column].dt, attr)

            dfs.append(df)

        Xt = pd.concat(dfs, axis=1)

        return Xt

class CountMatrixVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_features: int = 2 ** 15
    ) -> None:
        self.n_features = n_features
        self.vectorizers = {}
    @timeit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> 'CountMatrixVectorizer':
        self.columns = X.columns
        X.fillna('<missing>', inplace=True)
        for column in self.columns:
            hashing_vectorizer = HashingVectorizer(
                n_features = self.n_features
            )
            hashing_vectorizer.fit(X.loc[:,column].values)
            self.vectorizers[column] = hashing_vectorizer
        return self
    @timeit
    def transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        count_matrix = []
        X.fillna('<missing>', inplace=True)
        for column in self.columns:
            count_matrix.append(self.vectorizers[column].transform(X.loc[:,column].values))
        return hstack(tuple(count_matrix))
