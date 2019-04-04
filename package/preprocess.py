import datetime

from .CONSTANT import CATEGORY_PREFIX
from .CONSTANT import MULTI_CAT_PREFIX
from .CONSTANT import NUMERICAL_PREFIX
from .CONSTANT import TIME_PREFIX
from .utils import log
from .utils import timeit


@timeit
def clean_tables(tables):
    for tname in tables:
        log(f'cleaning table {tname}')
        clean_df(tables[tname])


@timeit
def clean_df(df):
    fillna(df)


@timeit
def fillna(df):
    for c in [c for c in df if c.startswith(NUMERICAL_PREFIX)]:
        df[c].fillna(-1, inplace=True)

    for c in [c for c in df if c.startswith(CATEGORY_PREFIX)]:
        df[c].fillna('0', inplace=True)

    for c in [c for c in df if c.startswith(TIME_PREFIX)]:
        df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)

    for c in [c for c in df if c.startswith(MULTI_CAT_PREFIX)]:
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
    for c in [c for c in df if c.startswith(CATEGORY_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x))

    for c in [c for c in df if c.startswith(MULTI_CAT_PREFIX)]:
        df[c] = df[c].apply(lambda x: int(x.split(',')[0]))
