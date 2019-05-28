from typing import Type
from typing import Union

from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BasePreprocessor
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class TimeVectorizer(BasePreprocessor):
    pass


class MultiValueCategoricalVectorizer(BasePreprocessor):
    _attributes = ['vectorizers_']

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        lowercase: bool = True,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

        self.lowercase = lowercase

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'MultiValueCategoricalVectorizer':
        v = TfidfVectorizer(dtype=self.dtype, lowercase=self.lowercase)

        self.vectorizers_ = [clone(v).fit(column) for column in X.T]

        return self

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        Xs = [
            self.vectorizers_[j].transform(
                column
            ) for j, column in enumerate(X.T)
        ]

        return hstack(Xs)
