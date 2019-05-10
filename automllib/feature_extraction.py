from typing import Any
from typing import List
from typing import Type
from typing import Union

import pandas as pd

from joblib import delayed
from joblib import effective_n_jobs
from joblib import Parallel
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils import gen_even_slices
from sklearn.utils import safe_indexing

from .base import BaseTransformer
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE


def multi_value_categorical_vectorize(
    X: TWO_DIM_ARRAY_TYPE,
    vectorizers: List[HashingVectorizer]
) -> TWO_DIM_ARRAY_TYPE:
    Xs = [vectorizers[j].transform(column) for j, column in enumerate(X.T)]

    return hstack(Xs)


class TimeVectorizer(BaseTransformer):
    _attributes = []

    def __init__(self, dtype: Union[str, Type] = None) -> None:
        super().__init__(dtype=dtype)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'TimeVectorizer':
        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = pd.DataFrame(X)
        dfs = []

        attributes = [
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

        for column in X:
            df = pd.DataFrame()

            for attr in attributes:
                df[f'{column}_{attr}'] = getattr(X[column].dt, attr)

            dfs.append(df)

        Xt = pd.concat(dfs, axis=1)
        _, n_features = Xt.shape

        self.logger_.info(
            f'{self.__class__.__name__} extracts {n_features} features.'
        )

        return Xt


class MultiValueCategoricalVectorizer(BaseTransformer):
    _attributes = ['vectorizers_']

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        lowercase: bool = True,
        n_features_per_column: int = 1_048_576,
        n_jobs: int = 1
    ) -> None:
        super().__init__(dtype=dtype)

        self.lowercase = lowercase
        self.n_features_per_column = n_features_per_column
        self.n_jobs = n_jobs

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'MultiValueCategoricalVectorizer':
        v = HashingVectorizer(
            lowercase=self.lowercase,
            n_features=self.n_features_per_column
        )

        self.vectorizers_ = [clone(v).fit(column) for column in X.T]

        return self

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        n_samples, _ = X.shape
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(multi_value_categorical_vectorize)
        result = parallel(
            func(
                safe_indexing(X, s), self.vectorizers_
            ) for s in gen_even_slices(n_samples, n_jobs)
        )

        return vstack(result)
