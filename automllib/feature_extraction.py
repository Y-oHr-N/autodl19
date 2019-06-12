from typing import Any
from typing import Dict
from typing import Type
from typing import Union

import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer

from .base import BasePreprocessor
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE


class TimeVectorizer(BasePreprocessor):
    _attributes = ['properties_']

    def __init__(
        self,
        dtype: Union[str, Type] = None,
        n_jobs: int = 1,
        verbose: int = 0
    ) -> None:
        super().__init__(dtype=dtype, n_jobs=n_jobs, verbose=verbose)

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'TimeVectorizer':
        secondsinminute = 60.0
        secondsinhour = 60.0 * secondsinminute
        secondsinday = 24.0 * secondsinhour
        secondsinweekday = 7.0 * secondsinday
        secondsinyear = 365.0 * secondsinday
        secondsinmonth = secondsinyear / 12.0

        self.properties_ = []

        for column in X.T:
            is_nan = pd.isnull(column)
            duration = (
                np.max(column[~is_nan]) - np.min(column[~is_nan])
            ) / np.timedelta64(1, 's')
            properties = []

            if duration > secondsinminute:
                properties.append('second')
            if duration > secondsinhour:
                properties.append('minute')
            if duration > secondsinday:
                properties.append('hour')
            if duration > secondsinweekday:
                properties.append('weekday')
            if duration > secondsinmonth:
                properties.append('day')
            if duration > secondsinyear:
                properties.extend(['month', 'quarter'])

            self.properties_.append(properties)

        return self

    def _parallel_transform(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        dtype = self.dtype

        if dtype is None:
            dtype = 'float64'

        n_samples, _ = X.shape
        Xs = []

        for j, column in enumerate(X.T):
            column = pd.Series(column)
            n_properties = len(self.properties_[j])
            Xt = np.empty((n_samples, 2 * n_properties), dtype=dtype)

            for k, attr in enumerate(self.properties_[j]):
                if attr in ['minute', 'second']:
                    period = 60.0
                if attr == 'hour':
                    period = 24.0
                elif attr == 'weekday':
                    period = 7.0
                elif attr == 'day':
                    period = column.dt.daysinmonth
                elif attr == 'month':
                    period = 12.0
                else:
                    period = 4.0

                theta = 2.0 * np.pi * getattr(column.dt, attr) / period

                Xt[:, 2 * k] = np.sin(theta)
                Xt[:, 2 * k + 1] = np.cos(theta)

            Xs.append(Xt)

        return np.concatenate(Xs, axis=1)


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
