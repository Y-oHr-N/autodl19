import pandas as pd

from scipy.stats import kurtosis

MAIN_TABLE_NAME = 'main'

CATEGORICAL_PREFIX = 'c_'
CATEGORICAL_TYPE = 'cat'

MULTI_VALUE_CATEGORICAL_DELIMITER = ','
MULTI_VALUE_CATEGORICAL_PREFIX = 'm_'
MULTI_VALUE_CATEGORICAL_TYPE = 'multi-cat'

NUMERICAL_PREFIX = 'n_'
NUMERICAL_TYPE = 'num'

TIME_PREFIX = 't_'
TIME_TYPE = 'time'

TYPE_MAP = {
    CATEGORICAL_TYPE: str,
    MULTI_VALUE_CATEGORICAL_TYPE: str,
    NUMERICAL_TYPE: float,
    TIME_TYPE: str
}

AGGREGATE_FUNCTIONS_MAP = {
    CATEGORICAL_TYPE: [
        pd.Series.nunique
    ],
    # MULTI_VALUE_CATEGORICAL_TYPE: [],
    NUMERICAL_TYPE: [
        'min',
        'max',
        'mean',
        # 'median',
        # 'sum',
        'var',
        'skew',
        kurtosis
    ],
    # TIME_TYPE: []
}
