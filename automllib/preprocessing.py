import datetime
import logging

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .constants import CATEGORICAL_PREFIX
from .constants import MULTI_VALUE_CATEGORICAL_PREFIX
from .constants import NUMERICAL_PREFIX
from .constants import TIME_PREFIX
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def clean_tables(tables):
    for tname in tables:
        logger.info(f'Clean table {tname}.')

        clean_df(tables[tname])


@timeit
def clean_df(df):
    categorical_feature_names = (
        df.columns[df.columns.str.startswith(CATEGORICAL_PREFIX)]
    )
    multi_value_categorical_feature_names = (
        df.columns[df.columns.str.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]
    )
    numerical_feature_names = (
        df.columns[df.columns.str.startswith(NUMERICAL_PREFIX)]
    )
    time_feature_names = (
        df.columns[df.columns.str.startswith(TIME_PREFIX)]
    )

    value = {}

    value.update({c: '0' for c in categorical_feature_names})
    value.update({c: '0' for c in multi_value_categorical_feature_names})
    value.update({c: -1 for c in numerical_feature_names})
    value.update(
        {c: datetime.datetime(1970, 1, 1) for c in time_feature_names}
    )

    df.fillna(value, inplace=True)


@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df)

    logger.info(f'The shape of X is {df.shape}.')


@timeit
def transform_datetime(df):
    time_feature_names = (
        df.columns[df.columns.str.startswith(TIME_PREFIX)]
    )

    df.drop(columns=time_feature_names, inplace=True)


@timeit
def transform_categorical_hash(df):
    categorical_feature_names = (
        df.columns[df.columns.str.startswith(CATEGORICAL_PREFIX)]
    )
    multi_value_categorical_feature_names = (
        df.columns[df.columns.str.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]
    )

    df[categorical_feature_names] = df[categorical_feature_names].astype(
        'uint'
    )

    for c in multi_value_categorical_feature_names:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))


class Clip(BaseEstimator, TransformerMixin):
    def __init__(self, low: float = 0.1, high: float = 99.9) -> None:
        self.low = low
        self.high = high

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'Clip':
        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if hasattr(X, 'values'):
            return X.clip(self.data_min_, self.data_max_, axis=1)
        else:
            return X.clip(self.data_min_, self.data_max_)
