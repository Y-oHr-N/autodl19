import logging

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array

from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit

logger = logging.getLogger(__name__)


class BaseSelector(BaseEstimator, ABC):
    @abstractmethod
    def __init__(self, **params: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'BaseSelector':
        pass

    @timeit
    def transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        X = check_array(
            X,
            dtype=None,
            estimator=self,
            force_all_finite='allow-nan'
        )
        _, n_features = X.shape
        n_selected_features = len(self.support_)
        n_dropped_features = n_features - n_selected_features

        logger.info(
            f'{self.__class__.__name__} selects {n_selected_features} '
            f'features and drops {n_dropped_features} features.'
        )

        return X[:, self.support_]


class DropDuplicates(BaseSelector, TransformerMixin):
    def __init__(self) -> None:
        pass

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropDuplicates':
        raise NotImplementedError()


class DropInvariant(BaseSelector, TransformerMixin):
    def __init__(self) -> None:
        pass

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropInvariant':
        X = check_array(
            X,
            dtype=None,
            estimator=self,
            force_all_finite='allow-nan'
        )

        self.support_ = np.array([
            j for j, column in enumerate(
                X.T
            ) if len(pd.unique(column)) > 1
        ])

        return self


class DropUniqueKey(BaseSelector, TransformerMixin):
    def __init__(self) -> None:
        pass

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'DropUniqueKey':
        X = check_array(
            X,
            dtype=None,
            estimator=self,
            force_all_finite='allow-nan'
        )
        n_samples = len(X)

        self.support_ = np.array([
            j for j, column in enumerate(
                X.T
            ) if len(pd.unique(column)) != n_samples
        ])

        return self


class NAProportionThreshold(BaseSelector, TransformerMixin):
    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    @timeit
    def fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
    ) -> 'NAProportionThreshold':
        X = check_array(
            X,
            dtype=None,
            estimator=self,
            force_all_finite='allow-nan'
        )
        n_samples = len(X)

        self.support_ = np.array([
            j for j, column in enumerate(
                X.T
            ) if pd.Series.count(column) >= (1.0 - self.threshold) * n_samples
        ])

        return self
