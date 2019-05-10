import logging

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

from joblib import dump
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.utils import safe_mask
from sklearn.utils.validation import check_is_fitted

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit


class BaseEstimator(SKLearnBaseEstimator, ABC):
    @property
    @abstractmethod
    def _attributes(self) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def __init__(self, **params: Any) -> None:
        pass

    @abstractmethod
    def _check_params(self) -> None:
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

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, self._attributes)

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
        **fit_params: Any
    ) -> 'BaseEstimator':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """

        self._check_params()

        X, y = self._check_X_y(X, y)

        self.logger_ = logging.getLogger(__name__)

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


class BaseSampler(BaseEstimator):
    _estimator_type = 'sampler'

    def _resample(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE,
    ) -> Tuple[TWO_DIM_ARRAY_TYPE, ONE_DIM_ARRAY_TYPE]:
        self._check_is_fitted()

        X, y = self._check_X_y(X, y)

        X = safe_indexing(X, self.sample_indices_)
        y = safe_indexing(y, self.sample_indices_)

        return X, y

    def fit_resample(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE,
        **fit_params: Any
    ) -> Tuple[TWO_DIM_ARRAY_TYPE, ONE_DIM_ARRAY_TYPE]:
        return self.fit(X, y, **fit_params)._resample(X, y)


class BaseTransformer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def __init__(self, dtype: Union[str, Type] = None):
        self.dtype = dtype

    @abstractmethod
    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        pass

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        self._check_is_fitted()

        X, _ = self._check_X_y(X)
        X = self._transform(X)

        if self.dtype is not None:
            X = X.astype(self.dtype)

        return X


class BaseSelector(BaseTransformer):
    @abstractmethod
    def _get_support(self) -> ONE_DIM_ARRAY_TYPE:
        pass

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        _, n_features = X.shape
        support = self.get_support()
        support = safe_mask(X, support)
        n_selected_features = np.sum(support)
        n_dropped_features = n_features - n_selected_features

        self.logger_.info(
            f'{self.__class__.__name__} selects {n_selected_features} '
            f'features and drops {n_dropped_features} features.'
        )

        return X[:, support]

    def get_support(self, indices: bool = False) -> ONE_DIM_ARRAY_TYPE:
        support = self._get_support()

        if indices:
            support = np.where(support)[0]

        return support
