import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


def parse_time(xtime: pd.Series):
    result = pd.DataFrame()

    s = pd.to_datetime(xtime, unit="s")

    result[f"{xtime.name}_unixtime"] = s.astype("int64") // 10 ** 9

    attrs = [
        # "year",
        # "weekofyear",
        "dayofyear",
        "quarter",
        "month",
        "day",
        "weekday",
        "hour",
        "minute",
        "second",
    ]

    for attr in attrs:
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
            period = 24.0
        elif attr in ["minute", "second"]:
            period = 60.0

        theta = 2.0 * np.pi * getattr(s.dt, attr) / period

        result[f"{xtime.name}_{attr}_sin"] = np.sin(theta)
        result[f"{xtime.name}_{attr}_cos"] = np.cos(theta)

    return result


class TypeAdapter(BaseEstimator, TransformerMixin):
    def __init__(self, primitive_cat):
        self.adapt_cols = primitive_cat.copy()

    def fit(self, X, y=None):
        cols_dtype = dict(zip(X.columns, X.dtypes))

        for key, dtype in cols_dtype.items():
            if dtype == np.dtype("object"):
                self.adapt_cols.append(key)

        return X

    def transform(self, X):
        X.loc[:, self.adapt_cols] = X.loc[:, self.adapt_cols].astype("category")

        return X
