import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


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


class MultiValueCategoricalVectorizer(BaseEstimator, TransformerMixin):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> 'MultiValueCategoricalVectorizer':
        self.vectorizers_ = []

        for column in X.T:
            vectorizer = HashingVectorizer()

            vectorizer.fit(column)

            self.vectorizers_.append(vectorizer)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        count_matrix = []

        for column, vectorizer in zip(X.T, self.vectorizers_):
            count_matrix.append(vectorizer.transform(column))

        return hstack(tuple(count_matrix))
