from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import List
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit


class BaseEstimator(SKLearnBaseEstimator, ABC):
    @abstractmethod
    def __init__(self, **params: Any) -> None:
        pass

    @abstractmethod
    def _check_params(self) -> None:
        pass

    @abstractmethod
    def _check_is_fitted(self) -> None:
        pass

    @abstractmethod
    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
        **fit_params: Any
    ) -> 'BaseEstimator':
        pass

    def _check_X_y(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
        accept_sparse: Union[str, bool, List[str]] = True,
        dtype: Union[str, Type, List[Type]] = None,
        force_all_finite: Union[str, bool] = 'allow-nan'
    ) -> TWO_DIM_ARRAY_TYPE:
        if y is None:
            X = check_array(
                X,
                accept_sparse=accept_sparse,
                estimator=self,
                dtype=dtype,
                force_all_finite=force_all_finite
            )
        else:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=accept_sparse,
                estimator=self,
                dtype=dtype,
                force_all_finite=force_all_finite
            )

        return X, y

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
        **fit_params: Any
    ) -> 'BaseEstimator':
        self._check_params()

        X, y = self._check_X_y(X, y)

        return self._fit(X, y, **fit_params)

    def to_pickle(
        self,
        filename: Union[str, Path],
        **kwargs: Any
    ) -> List[str]:
        """Persist an estimator object.

        Parameters
        ----------
        filename
            Path of the file in which it is to be stored.

        kwargs
            Other keywords passed to ``sklearn.externals.joblib.dump``.

        Returns
        -------
        filenames
            List of file names in which the data is stored.
        """

        self._check_is_fitted()

        return dump(self, filename, **kwargs)


class BaseTransformer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        pass

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

        X, _ = self._check_X_y(X)

        return self._transform(X)
