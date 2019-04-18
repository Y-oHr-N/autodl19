import datetime
import logging

from typing import Dict

from automllib.merge import Config
from automllib.constants import MAIN_TRAIN_TABLE_NAME
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
def feature_engineer(tables: Dict[str, pd.DataFrame], config: Config) -> None:
    for tname in tables:
        logger.info(f'feature engineering {tname}')
        transform_categorical_hash(tables[tname], tname, config)
        transform_datetime(tables[tname])
        logger.info(f'X.shape={tables[tname].shape}')

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
