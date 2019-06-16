from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from scipy.stats import ks_2samp
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

from .base import BaseSelector
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class DropCollinearFeatures(BaseSelector):
    """Drop collinear features.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import DropCollinearFeatures
    >>> sel = DropCollinearFeatures()
    >>> X = [[1, 1, 100], [2, 2, 10], [1, 1, 1], [1, 1, np.nan]]
    >>> Xt = sel.fit_transform(X)
    >>> Xt.shape
    (4, 2)
    """

    def __init__(self, threshold: float = 0.95, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'DropCollinearFeatures':
        X = X.astype('float64')

        self.corr_ = pd._libs.algos.nancorr(X)

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        triu = np.triu(self.corr_, k=1)
        triu = np.abs(triu)
        triu = np.nan_to_num(triu)

        return np.all(triu <= self.threshold, axis=0)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True}


class DropDriftFeatures(BaseSelector):
    """Drop drift features.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import DropDriftFeatures
    >>> sel = DropDriftFeatures()
    >>> X = [[1, 1, 100], [2, 2, 10], [1, 1, 1], [np.nan, 1, 1]]
    >>> X_test = [[1, 1000, 100], [2, 300, 10], [1, 100, 1], [1, 100, 1]]
    >>> Xt = sel.fit_transform(X, X_test=X_test)
    >>> Xt.shape
    (4, 2)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        random_state: Union[int, np.random.RandomState] = None,
        size: int = 100,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.alpha = alpha
        self.random_state = random_state
        self.size = size

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        X_test: TWO_DIM_ARRAYLIKE_TYPE = None
    ) -> 'DropDriftFeatures':
        if X_test is None:
            self.pvalues_ = None

            return self

        X_test, _ = self._check_X_y(X_test)
        random_state = check_random_state(self.random_state)

        self.pvalues_ = np.empty(self.n_features_)

        for j in range(self.n_features_):
            column = X[:, j]
            column_test = X_test[:, j]
            is_nan = pd.isnull(column)
            is_nan_test = pd.isnull(column_test)
            train = np.where(~is_nan)[0]
            train = random_state.choice(train, size=self.size)
            test = np.where(~is_nan_test)[0]
            test = random_state.choice(test, size=self.size)
            column = safe_indexing(column, train)
            column_test = safe_indexing(column_test, test)

            if issparse(column):
                column = np.ravel(column.toarray())

            if issparse(column_test):
                column_test = np.ravel(column_test.toarray())

            self.pvalues_[j] = ks_2samp(column, column_test).pvalue

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        if self.pvalues_ is None:
            return np.ones(self.n_features_, dtype=bool)

        return self.pvalues_ >= self.alpha

    def _more_tags(self) -> Dict[str, Any]:
        return {
            'allow_nan': True,
            'non_deterministic': True,
            'X_types': ['2darray', 'sparse']
        }


class FrequencyThreshold(BaseSelector):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import FrequencyThreshold
    >>> sel = FrequencyThreshold()
    >>> X = [[1, 1, 'Cat'], [2, 2, 'Cat'], [1, 3, 'Cat'], [1, 4, np.nan]]
    >>> Xt = sel.fit_transform(X)
    >>> Xt.shape
    (4, 1)
    """

    # TODO(Kon): Fix _check_X_y

    def __init__(
        self,
        max_frequency: Union[int, float] = 1.0,
        min_frequency: Union[int, float] = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.max_frequency = max_frequency
        self.min_frequency = min_frequency

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'FrequencyThreshold':
        self.n_samples_, _ = X.shape
        self.frequency_ = np.array([len(pd.unique(column)) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        max_frequency = self.max_frequency
        min_frequency = self.min_frequency

        if isinstance(max_frequency, float):
            max_frequency = int(max_frequency * self.n_samples_)

        if isinstance(min_frequency, float):
            min_frequency = int(min_frequency * self.n_samples_)

        return (self.frequency_ > min_frequency) \
            & (self.frequency_ < max_frequency)

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}


class NAProportionThreshold(BaseSelector):
    """

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.feature_selection import NAProportionThreshold
    >>> sel = NAProportionThreshold()
    >>> X = [[1, 1, 'Cat'], [2, 2, np.nan], [1, 1, np.nan], [1, 1, np.nan]]
    >>> Xt = sel.fit_transform(X)
    >>> Xt.shape
    (4, 2)
    """

    # TODO(Kon): Fix _check_X_y

    def __init__(self, threshold: float = 0.6, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

        self.threshold = threshold

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'NAProportionThreshold':
        self.n_samples_, _ = X.shape
        self.count_ = np.array([pd.Series.count(column) for column in X.T])

        return self

    def _get_support(self) -> ONE_DIM_ARRAYLIKE_TYPE:
        return self.count_ >= (1.0 - self.threshold) * self.n_samples_

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}
