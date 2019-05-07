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
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array

ONE_DIM_ARRAY_TYPE = Union[np.ndarray, pd.Series]
TWO_DIM_ARRAY_TYPE = Union[np.ndarray, spmatrix, pd.DataFrame]


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
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
        **fit_params: Any
    ) -> 'BaseEstimator':
        pass

    def _check_array(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        accept_sparse: bool = True,
        dtype: Union[str, Type] = None,
        force_all_finite: str = 'allow-nan'
    ) -> TWO_DIM_ARRAY_TYPE:
        return check_array(
            X,
            accept_sparse=accept_sparse,
            estimator=self,
            dtype=dtype,
            force_all_finite=force_all_finite
        )

    def to_pickle(
        self,
        filename: Union[str, Path],
        **kwargs: Any
    ) -> List[str]:
        """Persist an estimator object.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path of the file in which it is to be stored.

        kwargs : dict
            Other keywords passed to ``sklearn.externals.joblib.dump``.

        Returns
        -------
        filenames : list
            List of file names in which the data is stored.
        """

        self._check_is_fitted()

        return dump(self, filename, **kwargs)


class BaseTransformer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        pass
