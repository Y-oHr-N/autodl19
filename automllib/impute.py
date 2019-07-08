from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from .base import BasePreprocessor
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class ModifiedSimpleImputer(BasePreprocessor):
    """Imputation transformer for completing missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from automllib.impute import ModifiedSimpleImputer
    >>> imp = ModifiedSimpleImputer()
    >>> X = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
    >>> X_test = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> imp.fit(X)
    ModifiedSimpleImputer(...)
    >>> imp.transform(X_test)
    array([[ 7. ,  2. ,  3. ],
           [ 4. ,  3.5,  6. ],
           [10. ,  3.5,  9. ]])
    """

    def __init__(
        self,
        copy: bool = True,
        dtype: Union[str, Type] = None,
        fill_value: Any = None,
        n_jobs: int = 1,
        strategy: str = 'mean',
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

        self.copy = copy
        self.fill_value = fill_value
        self.strategy = strategy

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'ModifiedSimpleImputer':
        dtype = self.dtype
        fill_value = self.fill_value

        if X.dtype.kind not in ('f', 'i', 'u'):
            if self.strategy in ['mean', 'min']:
                dtype = 'float64'
            else:
                dtype = X.dtype

        if self.fill_value is None:
            if X.dtype.kind in ('f', 'i', 'u'):
                fill_value = 0
            else:
                fill_value = 'missing_value'

        if self.strategy == 'constant':
            self.statistics_ = np.full(
                self.n_features_,
                fill_value,
                dtype=dtype
            )

        elif self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0, dtype=dtype)

        elif self.strategy == 'min':
            self.statistics_ = np.empty(self.n_features_, dtype=dtype)

            for j, column in enumerate(X.T):
                is_nan = pd.isnull(column)
                self.statistics_[j] = np.min(column[~is_nan])

        else:
            raise ValueError(f'Invalid strategy: {self.strategy}.')

        return self

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'X_types': ['2darray', 'str']}

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype

        if X.dtype.kind not in ('f', 'i', 'u'):
            dtype = X.dtype

        X = X.astype(dtype, copy=self.copy)

        for j, column in enumerate(X.T):
            is_nan = pd.isnull(column)
            X[is_nan, j] = self.statistics_[j]

        return X
