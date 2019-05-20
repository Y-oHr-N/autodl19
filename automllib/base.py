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
import pandas as pd

from joblib import dump
from sklearn.base import BaseEstimator as SKLearnBaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import safe_indexing
from sklearn.utils import safe_mask
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted

from .utils import Timeit


class BaseEstimator(SKLearnBaseEstimator, ABC):
    @property
    @abstractmethod
    def _attributes(self) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose

    @abstractmethod
    def _check_params(self) -> None:
        pass

    @abstractmethod
    def _fit(
        self,
        X: Any,
        y: pd.Series = None,
        **fit_params: Any
    ) -> 'BaseEstimator':
        pass

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
        X: Any,
        y: pd.Series = None,
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

        self._check_is_fitted()

        return dump(self, filename, **kwargs)


class BaseSampler(BaseEstimator):
    _estimator_type = 'sampler'

    def _resample(
        self,
        X: Any,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        n_input_samples = _num_samples(X)
        X = safe_indexing(X, self.sample_indices_)
        y = safe_indexing(y, self.sample_indices_)
        n_output_samples = len(self.sample_indices_)

        self.logger_.info(
            f'{self.__class__.__name__} selects {n_output_samples} samples '
            f'and drops {n_input_samples - n_output_samples} samples.'
        )

        return X, y

    def fit_resample(
        self,
        X: Any,
        y: pd.Series,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.Series]:
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

        func = self.timeit_(self._resample)

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
    def _transform(self, X: Any) -> pd.DataFrame:
        pass

    def transform(self, X: Any) -> pd.DataFrame:
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

        func = self.timeit_(self._transform)

        X = func(X)

        if self.dtype is not None:
            X = X.astype(self.dtype)

        return X


class BaseSelector(BaseTransformer):
    @abstractmethod
    def _get_support(self) -> np.ndarray:
        pass

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        _, n_input_features = X.shape
        support = self.get_support()
        support = safe_mask(X, support)
        n_output_features = np.sum(support)

        self.logger_.info(
            f'{self.__class__.__name__} selects {n_output_features} '
            f'features and drops {n_input_features - n_output_features} '
            f'features.'
        )

        return X.iloc[:, support]

    def get_support(self, indices: bool = False) -> np.ndarray:
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
