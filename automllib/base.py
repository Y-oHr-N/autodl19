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
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import Timeit


class BaseEstimator(SKLearnBaseEstimator, ABC):
    @property
    @abstractmethod
    def _attributes(self) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def __init__(self, validate: bool = True, verbose: int = 0) -> None:
        self.validate = validate
        self.verbose = verbose

    @abstractmethod
    def _check_params(self) -> None:
        pass

    @abstractmethod
    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
        *args: Any,
        **kwargs: Any
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
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None,
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

        if self.validate:
            self._check_params()

            X, y = self._check_X_y(X, y)

        self.logger_ = self._get_logger()
        self.timeit_ = Timeit(self.logger_)

        func = self.timeit_(self._fit)

        return func(X, y, *args, **kwargs)

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
            Other keywords passed to ``joblib.dump``.

        Returns
        -------
        filenames
            List of file names in which the data is stored.
        """

        if self.validate:
            self._check_is_fitted()

        return dump(self, filename, **kwargs)


class BaseSampler(BaseEstimator):
    _estimator_type = 'sampler'

    def _resample(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE,
    ) -> Tuple[TWO_DIM_ARRAY_TYPE, ONE_DIM_ARRAY_TYPE]:
        if self.validate:
            self._check_is_fitted()

            X, y = self._check_X_y(X, y)

        X = safe_indexing(X, self.sample_indices_)
        y = safe_indexing(y, self.sample_indices_)
        n_input_samples = _num_samples(X)
        n_output_samples = len(self.sample_indices_)

        self.logger_.info(
            f'{self.__class__.__name__} selects {n_output_samples} samples '
            f'and drops {n_input_samples - n_output_samples} samples.'
        )

        return X, y

    def fit_resample(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE,
        **fit_params: Any
    ) -> Tuple[TWO_DIM_ARRAY_TYPE, ONE_DIM_ARRAY_TYPE]:
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

        self.fit(X, y, **fit_params)

        func = self.timeit_(self._resample)

        return func(X, y)


class BaseTransformer(BaseEstimator, TransformerMixin):
    @abstractmethod
    def __init__(
        self,
        dtype: Union[str, Type] = None,
        validate: bool = True,
        verbose: int = 0
    ) -> None:
        super().__init__(validate=validate, verbose=verbose)

        self.dtype = dtype

    @abstractmethod
    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        pass

    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
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

        if self.validate:
            self._check_is_fitted()

            X, _ = self._check_X_y(X)

        func = self.timeit_(self._transform)

        X = func(X)

        if self.dtype is not None:
            X = X.astype(self.dtype)

        return X


class BaseSelector(BaseTransformer):
    @abstractmethod
    def _get_support(self) -> ONE_DIM_ARRAY_TYPE:
        pass

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        _, n_input_features = X.shape
        support = self.get_support()
        support = safe_mask(X, support)
        n_output_features = np.sum(support)

        self.logger_.info(
            f'{self.__class__.__name__} selects {n_output_features} '
            f'features and drops {n_input_features - n_output_features} '
            f'features.'
        )

        return X[:, support]

    def get_support(self, indices: bool = False) -> ONE_DIM_ARRAY_TYPE:
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

        support = self._get_support()

        if indices:
            support = np.where(support)[0]

        return support
