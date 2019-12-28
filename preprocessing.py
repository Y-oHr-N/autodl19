import logging

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
import datetime

try:  # scikit-learn<=0.21
    from sklearn.feature_selection.from_model import _calculate_threshold
    from sklearn.feature_selection.from_model import _get_feature_importances
except ImportError:
    from sklearn.feature_selection._from_model import _calculate_threshold
    from sklearn.feature_selection._from_model import _get_feature_importances

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)

logger.setLevel(logging.INFO)


class Astype(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols: List[str], numerical_cols: List[str]) -> None:
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Astype":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xt = X.copy()

        if self.categorical_cols:
            Xt[self.categorical_cols] = Xt[self.categorical_cols].astype("category")

        if self.numerical_cols:
            Xt[self.numerical_cols] = Xt[self.numerical_cols].astype("float32")

        return Xt


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, dtype: str = "float32", encode: bool = False) -> None:
        self.dtype = dtype
        self.encode = encode

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
                # if s.dt.dayofyear.nunique() > 1:
                #     attrs.append("dayofyear")
                # if s.dt.weekofyear.nunique() > 1:
                #     attrs.append("weekofyear")
                # if s.dt.quarter.nunique() > 1:
                #     attrs.append("quarter")
                if s.dt.month.nunique() > 1:
                    attrs.append("month")
            if duration >= 2.0 * secondsinmonth and s.dt.day.nunique() > 1:
                attrs.append("day")
            if duration >= 2.0 * secondsinweekday and s.dt.weekday.nunique() > 1:
                attrs.append("weekday")
            if duration >= 2.0 * secondsinday and s.dt.hour.nunique() > 1:
                attrs.append("hour")
            # if duration >= 2.0 * secondsinhour and s.dt.minute.nunique() > 1:
            #     attrs.append("minute")
            # if duration >= 2.0 * secondsinminute and s.dt.second.nunique() > 1:
            #     attrs.append("second")

            self.attributes_[col] = attrs

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        for col in X:
            s = X[col]

            unixtime = 1e-09 * s.astype("int64")
            unixtime = unixtime.astype(self.dtype)

            Xt[col] = unixtime

            for attr in self.attributes_[col]:
                x = getattr(s.dt, attr)

                if not self.encode:
                    x = x.astype("category")

                    Xt["{}_{}".format(col, attr)] = x

                    continue

                if attr == "dayofyear":
                    period = np.where(s.dt.is_leap_year, 366.0, 365.0)
                elif attr == "weekofyear":
                    period = 52.1429
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
                sin_theta = np.sin(theta)
                sin_theta = sin_theta.astype(self.dtype)
                cos_theta = np.cos(theta)
                cos_theta = cos_theta.astype(self.dtype)

                Xt["{}_{}_sin".format(col, attr)] = sin_theta
                Xt["{}_{}_cos".format(col, attr)] = cos_theta

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
        X = pd.DataFrame(X)

        return X.clip(self.data_min_, self.data_max_, axis=1)


class ModifiedSelectFromModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: bool = True,
        threshold: Optional[Union[float, str]] = None,
        train_size: float = 0.75,
    ) -> None:
        self.estimator = estimator
        self.random_state = random_state
        self.shuffle = shuffle
        self.threshold = threshold
        self.train_size = train_size

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params: Any
    ) -> "ModifiedSelectFromModel":
        if self.train_size < 1.0:
            X, _, y, _ = train_test_split(
                X,
                y,
                random_state=self.random_state,
                shuffle=self.shuffle,
                train_size=self.train_size,
            )

        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y, **fit_params)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)

        feature_importances = _get_feature_importances(self.estimator_)
        threshold = _calculate_threshold(
            self.estimator_, feature_importances, self.threshold
        )
        cols = feature_importances >= threshold
        _, n_features = X.shape
        n_dropped_features = n_features - np.sum(cols)

        logger.info("{} features are dropped.".format(n_dropped_features))

        return X.loc[:, cols]


class Profiler(BaseEstimator, TransformerMixin):
    def __init__(self, label_col: str = "label") -> None:
        self.label_col = label_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Profiler":
        data = pd.DataFrame(X)

        if y is not None:
            kwargs = {self.label_col: y}
            data = X.assign(**kwargs)

        summary = data.describe(include="all")

        with pd.option_context("display.max_columns", None, "display.precision", 3):
            logger.info(summary)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)


class TargetShiftFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        shift_range: List[int] = [1],
        pred_time_diff: int = 0,
        primary_id: List[str] = None,
        time_col: str = None,
    ) -> None:
        self.shift_range = shift_range
        self.pred_time_diff = pred_time_diff
        self.primary_id = primary_id
        self.time_col = time_col
        self.time_delta = datetime.timedelta(
            seconds=max(self.shift_range) * self.pred_time_diff
        )

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "TargetShiftFeatures":
        self.X = X[[self.time_col] + self.primary_id]
        self.X["target"] = y

    def transform(self, X: pd.DataFrame, istrain: bool = True) -> pd.DataFrame:
        if istrain:
            if self.primary_id:
                grouped = self.X.groupby(self.primary_id)
            else:
                grouped = self.X
            for i in self.shift_range:
                X[f"target_{i}_shift"] = grouped["target"].shift(i)

        else:
            X_tmp = X[[self.time_col] + self.primary_id]
            target_time_col = X_tmp.iloc[0, :][self.time_col]
            X_tmp["target"] = np.nan
            X_tmp = pd.concat(
                [
                    self.X[self.X[self.time_col] >= target_time_col - self.time_delta],
                    X_tmp,
                ],
                axis=0,
            )
            if self.primary_id:
                grouped = X_tmp.groupby(self.primary_id)
            else:
                grouped = X_tmp
            for i in self.shift_range:
                X_tmp[f"target_{i}_shift"] = grouped["target"].shift(i)
                X[f"target_{i}_shift"] = X_tmp[X_tmp[self.time_col] == target_time_col][
                    f"target_{i}_shift"
                ]
        for key in self.primary_id:
            X[key] = X[key].astype("category")
        return X

    def update(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = X[[self.time_col] + self.primary_id]
        X["target"] = y
        self.X = pd.concat([self.X, X], axis=0).reset_index(drop=True)


def get_pred_time_diff(pred_timestamp, time_col):
    pred_time_diff = pred_timestamp[time_col][1] - pred_timestamp[time_col][0]
    pred_time_diff = pred_time_diff.total_seconds()
    return pred_time_diff


def get_time_shift_range(pred_timestamp, time_col):
    secondsinminute = 60.0
    secondsinhour = 60.0 * secondsinminute
    secondsinday = 24.0 * secondsinhour
    secondsinmonth = 28.0 * secondsinday
    pred_time_diff = get_pred_time_diff(pred_timestamp, time_col)
    if pred_time_diff >= secondsinmonth:
        time_shift_range = [1, 2, 6, 12]
    elif pred_time_diff >= secondsinday:
        time_shift_range = [1, 2, 7, 28]
    elif pred_time_diff >= secondsinhour:
        time_shift_range = [1, 2, 12, 24]
    elif pred_time_diff >= 10.0 * secondsinminute:
        time_shift_range = [1, 2, 6]
    elif pred_time_diff >= secondsinminute:
        time_shift_range = [1, 2, 10, 60]
    else:
        time_shift_range = [1, 2, 3]
    return time_shift_range
