import datetime
import logging

from .constants import CATEGORICAL_PREFIX
from .constants import MULTI_VALUE_CATEGORICAL_PREFIX
from .constants import NUMERICAL_PREFIX
from .constants import TIME_PREFIX
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def clean_tables(tables):
    for tname in tables:
        logger.info(f'cleaning table {tname}')
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CATEGORICAL_PREFIX)]:
        df[c].fillna('0', inplace=True)

    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]:
        df[c].fillna('0', inplace=True)


@timeit
def feature_engineer(df, config):
    transform_categorical_hash(df)
    transform_datetime(df, config)


@timeit
def transform_datetime(df, config):
    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df.drop(c, axis=1, inplace=True)


@timeit
def transform_categorical_hash(df):
    for c in [c for c in df if c.startswith(CATEGORICAL_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    for c in [c for c in df if c.startswith(MULTI_VALUE_CATEGORICAL_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
