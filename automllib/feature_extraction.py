from typing import Any
from typing import Dict
from typing import Type
from typing import Union

from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer

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
        n_features: int = 1_048_576,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

        self.lowercase = lowercase
        self.n_features = n_features

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'MultiValueCategoricalVectorizer':
        dtype = self.dtype

        if dtype is None:
            dtype = 'float64'

        v = HashingVectorizer(
            dtype=self.dtype,
            lowercase=self.lowercase,
            n_features=self.n_features
        )

        self.vectorizers_ = [clone(v).fit(column) for column in X.T]

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'X_types': ['2darray', 'str']}

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
