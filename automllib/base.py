import logging
import pathlib

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from joblib import delayed
from joblib import dump
from joblib import effective_n_jobs
from joblib import Parallel
from scipy.sparse import issparse
from scipy.sparse import spmatrix
from scipy.sparse import vstack
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import gen_even_slices
from sklearn.utils import safe_indexing
from sklearn.utils import safe_mask
from sklearn.utils.validation import check_is_fitted

from .utils import Timeit

ONE_DIM_ARRAYLIKE_TYPE = Union[np.ndarray, pd.Series]
TWO_DIM_ARRAYLIKE_TYPE = Union[np.ndarray, spmatrix, pd.DataFrame]


class BaseEstimator(SKLearnBaseEstimator, ABC):
    @abstractmethod
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose

    @abstractmethod
    def _check_params(self) -> None:
        pass

    @abstractmethod
    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        **fit_params: Any
    ) -> 'BaseEstimator':
        pass

    def _check_X_y(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        n_features: int = None
    ) -> Tuple[TWO_DIM_ARRAYLIKE_TYPE, ONE_DIM_ARRAYLIKE_TYPE]:
        tags = self._get_tags()
        accept_sparse = 'sparse' in tags['X_types']

        if 'str' in tags['X_types'] or 'categorical' in tags['X_types']:
            dtype = None
        else:
            dtype = 'numeric'

        if tags['allow_nan']:
            force_all_finite = 'allow-nan'
        else:
            force_all_finite = True

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

        if n_features is not None:
            _, n_input_features = X.shape

            if n_input_features != n_features:
                raise ValueError(f'Invalid data: shape={X.shape}')

        return X, y

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, 'n_features_')

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)

        if self.verbose > 1:
            logger.setLevel(logging.DEBUG)
        elif self.verbose > 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        return logger

    def fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        *args: Any,
        **kwargs: Any
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

        tags = self._get_tags()
        logger = self._get_logger()
        timeit = Timeit(logger=logger)
        func = timeit(self._fit)

        if not tags['no_validation']:
            X, y = self._check_X_y(X, y)

        _, self.n_features_ = X.shape

        return func(X, y, *args, **kwargs)

    def to_pickle(
        self,
        filename: Union[str, pathlib.Path],
        **kwargs: Any
    ) -> List[str]:
        """Persist an estimator object.

        Parameters
        ----------
        filename
            Path of the file in which it is to be stored.

        kwargs
            Other keywords passed to ``joblib.dump``.

        Returns
        -------
        filenames
            List of file names in which the data is stored.
        """

        self._check_is_fitted()

        return dump(self, filename, **kwargs)


class BaseSampler(BaseEstimator):
    _estimator_type = 'sampler'

    @property
    @abstractmethod
    def _sampling_type(self) -> str:
        pass

    def _resample(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
    ) -> Tuple[TWO_DIM_ARRAYLIKE_TYPE, ONE_DIM_ARRAYLIKE_TYPE]:
        logger = self._get_logger()
        n_input_samples, _ = X.shape
        n_output_samples, = self.sample_indices_.shape
        X = safe_indexing(X, self.sample_indices_)
        y = safe_indexing(y, self.sample_indices_)

        logger.info(
            f'{self.__class__.__name__} selects {n_output_samples} samples '
            f'and drops {n_input_samples - n_output_samples} samples.'
        )

        return X, y

    def fit_resample(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[TWO_DIM_ARRAYLIKE_TYPE, ONE_DIM_ARRAYLIKE_TYPE]:
        """Fit the model, then resample the data.

        Parameters
        ----------
        X
            Training Data.

        y
            Target.

        Returns
        -------
        X_res
            Resampled data.

        y_res
            Resampled target.
        """

        self.fit(X, y, *args, **kwargs)

        logger = self._get_logger()
        timeit = Timeit(logger)
        func = timeit(self._resample)

        return func(X, y)


class BaseTransformer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def __init__(
        self,
        dtype: Union[str, Type] = None,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.dtype = dtype

    @abstractmethod
    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        pass

    def transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        """Transform the data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        Xt
            Transformed data.
        """

        self._check_is_fitted()

        tags = self._get_tags()
        logger = self._get_logger()
        timeit = Timeit(logger)
        func = timeit(self._transform)

        if not tags['no_validation']:
            X, _ = self._check_X_y(X, n_features=self.n_features_)

        X = func(X)

        return X


class BasePreprocessor(BaseTransformer):
    @abstractmethod
    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.dtype = dtype
        self.n_jobs = n_jobs

    @abstractmethod
    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        pass

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        n_samples, _ = X.shape
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(self._parallel_transform)
        iterable = gen_even_slices(n_samples, n_jobs)
        Xs = parallel(func(X[s]) for s in iterable)

        if np.any([issparse(Xt) for Xt in Xs]):
            X = vstack(Xs)
        else:
            X = np.concatenate(Xs)

        if self.dtype is not None and self.dtype != X.dtype:
            X = X.astype(dtype=self.dtype)

        return X


class BaseSelector(BaseTransformer):
    @abstractmethod
    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        pass

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        logger = self._get_logger()
        _, n_input_features = X.shape
        support = self._get_support()
        n_output_features = np.sum(support)
        support = safe_mask(X, support)

        logger.info(
            f'{self.__class__.__name__} selects {n_output_features} '
            f'features and drops {n_input_features - n_output_features} '
            f'features.'
        )

        return X[:, support]

    def get_support(self, indices: bool = False) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Get a mask of the features selected.

        Parameters
        ----------
        indices
           If True, an array of integers is returned.

        Returns
        -------
        support
            Mask.
        """

        self._check_is_fitted()

        support = self._get_support()

        if indices:
            support = np.where(support)[0]

        return support
