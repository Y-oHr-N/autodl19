import datetime
import logging

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from .utils import get_categorical_columns
from .utils import get_multi_value_categorical_columns
from .utils import get_numerical_columns
from .utils import get_time_columns
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def clean_tables(tables):
    for tname in tables:
        logger.info(f'Clean table {tname}.')

        clean_df(tables[tname])


@timeit
def clean_df(df):
    c_feature_names = get_categorical_columns(df)
    m_feature_names = get_multi_value_categorical_columns(df)
    n_feature_names = get_numerical_columns(df)
    t_feature_names = get_time_columns(df)

    value = {}

    value.update({name: '0' for name in c_feature_names})
    value.update({name: '0' for name in m_feature_names})
    value.update({name: -1 for name in n_feature_names})
    value.update(
        {name: datetime.datetime(1970, 1, 1) for name in t_feature_names}
    )

    df.fillna(value, inplace=True)


@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df)

    logger.info(f'The shape of X is {df.shape}.')


@timeit
def transform_datetime(df):
    t_feature_names = get_time_columns(df)

    df.drop(columns=t_feature_names, inplace=True)


@timeit
def transform_categorical_hash(df):
    c_feature_names = get_categorical_columns(df)
    m_feature_names = get_multi_value_categorical_columns(df)

    df[c_feature_names] = df[c_feature_names].astype('uint')

    for name in m_feature_names:
        df[name] = df[name].apply(lambda x: int(x.split(',')[0]))


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
