from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin

try:  # scikit-learn<=0.21
    from sklearn.feature_selection.from_model import _calculate_threshold
    from sklearn.feature_selection.from_model import _get_feature_importances
except ImportError:
    from sklearn.feature_selection._from_model import _calculate_threshold
    from sklearn.feature_selection._from_model import _get_feature_importances


class TypeAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols, numerical_cols, time_cols):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.time_cols = time_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for key in X.columns:
            if key in self.categorical_cols:
                X[key] = X[key].astype("category")
            elif key in self.numerical_cols:
                X[key] = X[key].astype("float32")
            elif key in self.time_cols:
                X[key] = pd.to_datetime(X[key], unit="s")

        return X


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CalendarFeatures":
        X = pd.DataFrame(X)

        secondsinminute = 60.0
        secondsinhour = 60.0 * secondsinminute
        secondsinday = 24.0 * secondsinhour
        secondsinweekday = 7.0 * secondsinday
        secondsinmonth = 30.4167 * secondsinday
        secondsinyear = 12.0 * secondsinmonth

        self.attributes_: Dict[str, List[str]] = {}

        for col in X:
            s = X[col]
            duration = s.max() - s.min()
            duration = duration.total_seconds()
            attrs = []

            if duration >= 2.0 * secondsinyear:
                if s.dt.dayofyear.nunique() > 1:
                    attrs.append("dayofyear")
                if s.dt.quarter.nunique() > 1:
                    attrs.append("quarter")
                if s.dt.month.nunique() > 1:
                    attrs.append("month")
            if duration >= 2.0 * secondsinmonth and s.dt.day.nunique() > 1:
                attrs.append("day")
            if duration >= 2.0 * secondsinweekday and s.dt.weekday.nunique() > 1:
                attrs.append("weekday")
            if duration >= 2.0 * secondsinday and s.dt.hour.nunique() > 1:
                attrs.append("hour")
            # if duration >= 2.0 * secondsinhour \
            #         and s.dt.minute.nunique() > 1:
            #     attrs.append("minute")
            # if duration >= 2.0 * secondsinminute \
            #         and s.dt.second.nunique() > 1:
            #     attrs.append("second")

            self.attributes_[col] = attrs

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        for col in X:
            s = X[col]
            Xt[col] = 1e-09 * s.astype("int64")

            for attr in self.attributes_[col]:
                x = getattr(s.dt, attr)

                if attr == "dayofyear":
                    period = np.where(s.dt.is_leap_year, 366.0, 365.0)
                elif attr == "quarter":
                    period = 4.0
                elif attr == "month":
                    period = 12.0
                elif attr == "day":
                    period = s.dt.daysinmonth
                elif attr == "weekday":
                    period = 7.0
                elif attr == "hour":
                    x += s.dt.minute / 60.0 + s.dt.second / 60.0
                    period = 24.0
                elif attr in ["minute", "second"]:
                    period = 60.0

                theta = 2.0 * np.pi * x / period

                Xt["{}_{}_sin".format(s.name, attr)] = np.sin(theta)
                Xt["{}_{}_cos".format(s.name, attr)] = np.cos(theta)

        return Xt.astype(self.dtype)


class ClippedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, high: float = 99.0, low: float = 1.0) -> None:
        self.high = high
        self.low = low

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClippedFeatures":
        self.data_min_, self.data_max_ = np.nanpercentile(
            X, [self.low, self.high], axis=0
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)

        return X.clip(self.data_min_, self.data_max_, axis=1)


class ModifiedSelectFromModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        threshold: Optional[Union[float, str]] = None
    ):
        self.estimator = estimator
        self.threshold = threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **fit_params: Any
    ) -> 'ModifiedSelectFromModel':
        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y, **fit_params)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)

        feature_importances = _get_feature_importances(self.estimator_)
        threshold = _calculate_threshold(
            self.estimator_,
            feature_importances,
            self.threshold
        )
        cols = feature_importances >= threshold

        return X.loc[:, cols]
