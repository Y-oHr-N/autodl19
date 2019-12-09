from typing import Optional

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class TypeAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, primitive_cat):
        self.adapt_cols = primitive_cat.copy()

    def fit(self, X, y=None):
        cols_dtype = dict(zip(X.columns, X.dtypes))

        for key, dtype in cols_dtype.items():
            if dtype == np.dtype("object"):
                self.adapt_cols.append(key)

        return self

    def transform(self, X):
        for key in X.columns:
            if key in self.adapt_cols:
                X[key] = X[key].astype("category")

        return X


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CalendarFeatures":
        secondsinminute = 60.0
        secondsinhour = 60.0 * secondsinminute
        secondsinday = 24.0 * secondsinhour
        secondsinweekday = 7.0 * secondsinday
        secondsinmonth = 30.4167 * secondsinday
        secondsinyear = 12.0 * secondsinmonth

        self.attributes_ = {}

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
        Xt = pd.DataFrame()

        for col in X:
            s = X[col]
            # Xt[col] = 1e-09 * s.astype('int64')

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

        return Xt


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
        return X.clip(self.data_min_, self.data_max_, axis=1)
