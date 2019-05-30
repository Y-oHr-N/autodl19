import logging

from collections import defaultdict
from collections import deque
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd

from scipy.sparse import spmatrix
from scipy.stats import kurtosis

from .base import BaseTransformer
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE

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


def get_feature_names_by_prefix(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    prefix: str
) -> ONE_DIM_ARRAYLIKE_TYPE:
    is_startwith = X.columns.str.startswith(prefix)
    n_features = is_startwith.sum()

    logger.info(f'Number of features starting with {prefix} is {n_features}.')

    return X.columns[is_startwith]


def get_categorical_feature_names(X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
    return get_feature_names_by_prefix(X, CATEGORICAL_PREFIX)


def get_multi_value_categorical_feature_names(
    X: TWO_DIM_ARRAYLIKE_TYPE
) -> ONE_DIM_ARRAYLIKE_TYPE:
    return get_feature_names_by_prefix(X, MULTI_VALUE_CATEGORICAL_PREFIX)


def get_numerical_feature_names(X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
    return get_feature_names_by_prefix(X, NUMERICAL_PREFIX)


def get_time_feature_names(X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
    return get_feature_names_by_prefix(X, TIME_PREFIX)


logger = logging.getLogger(__name__)


def join(u, v, v_name, key, type_):
    if type_.split("_")[2] == 'many':
        columns = v.columns.drop(key)
        func = aggregate_functions(columns)
        v = v.groupby(key).agg(func)
        v.columns = v.columns.map(
            lambda a: f"{NUMERICAL_PREFIX}{a[1].upper()}({a[0]})"
        )
    else:
        v = v.set_index(key)

    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)


def temporal_join(u, v, v_name, key, time_col):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    tmp_u = tmp_u.sort_values(time_col)
    tmp_u = tmp_u.groupby(key).ffill()

    tmp_u.columns = tmp_u.columns.map(
        lambda a: f"{a}_NEAREST({v_name})" if not (a == key or a == time_col) else a
    )
    tmp_u = tmp_u.drop(columns=[key, time_col])

    if tmp_u.empty:
        logger.info('Return u because temp_u is empty.')
        return u

    return pd.concat([u, tmp_u.loc['u']], axis=1, join_axes=[u.index])


def dfs(u_name, config, tables, graph):
    u = tables[u_name]

    logger.info(f'Enter {u_name}.')

    for edge in graph[u_name]:
        v_name = edge['to']

        if config['tables'][v_name]['depth'] <= config['tables'][u_name]['depth']:
            continue

        v = dfs(v_name, config, tables, graph)
        key = edge['key']
        type_ = edge['type']

        if config['time_col'] not in u and config['time_col'] in v:
            continue

        if config['time_col'] in u and config['time_col'] in v:
            logger.info(f'Temporal Join {u_name} <--{type_}--t {v_name}.')
            u = temporal_join(u, v, v_name, key, config['time_col'])
        else:
            logger.info(f'Join {u_name} <--{type_}--nt {v_name}.')
            u = join(u, v, v_name, key, type_)

        del v

    logger.info(f'Leave {u_name}.')

    return u


def aggregate_functions(
    X: TWO_DIM_ARRAYLIKE_TYPE
) -> Dict[str, List[Union[str, Callable]]]:
    AFS_MAP = {
        CATEGORICAL_TYPE: [
            'count',
            'last',
            pd.Series.nunique
        ],
        MULTI_VALUE_CATEGORICAL_TYPE: [
            'count',
            'last',
            pd.Series.nunique
        ],
        NUMERICAL_TYPE: [
            'count',
            'last',
            'min',
            'max',
            'mean',
            # 'median',
            'sum',
            'std',
            'skew',
            kurtosis
        ],
        TIME_TYPE: [
            'count',
            'last'
        ]
    }
    func = {}

    c_feature_names = get_categorical_feature_names(X)
    m_feature_names = get_multi_value_categorical_feature_names(X)
    n_feature_names = get_numerical_feature_names(X)
    t_feature_names = get_time_feature_names(X)

    func.update({name: AFS_MAP[CATEGORICAL_TYPE] for name in c_feature_names})
    func.update({name: AFS_MAP[MULTI_VALUE_CATEGORICAL_TYPE] for name in m_feature_names})
    func.update({name: AFS_MAP[NUMERICAL_TYPE] for name in n_feature_names})
    func.update({name: AFS_MAP[TIME_TYPE] for name in t_feature_names})

    return func


class TableJoiner(BaseTransformer):
    _attributes = ['config_', 'graph_']

    def __init__(
        self,
        info: Dict[str, Any],
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE],
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.info = info
        self.related_tables = related_tables

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'TableJoiner':
        self.graph_ = defaultdict(list)

        for rel in self.info['relations']:
            ta = rel['table_A']
            tb = rel['table_B']

            self.graph_[ta].append({
                'to': tb,
                'key': rel['key'],
                'type': rel['type']
            })

            self.graph_[tb].append({
                'to': ta,
                'key': rel['key'],
                'type': '_'.join(rel['type'].split('_')[::-1])
            })

        self.config_ = self.info.copy()
        self.config_['tables'] = {}

        for tname, ttype in self.info['tables'].items():
            self.config_['tables'][tname] = {}
            self.config_['tables'][tname]['type'] = ttype

        self.config_['tables'][MAIN_TABLE_NAME]['depth'] = 0

        queue = deque([MAIN_TABLE_NAME])

        while queue:
            u_name = queue.popleft()

            for edge in self.graph_[u_name]:
                v_name = edge['to']

                if 'depth' not in self.config_['tables'][v_name]:
                    self.config_['tables'][v_name]['depth'] = \
                        self.config_['tables'][u_name]['depth'] + 1

                    queue.append(v_name)

        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        Xs = self.related_tables.copy()
        Xs[MAIN_TABLE_NAME] = X

        return dfs(
            MAIN_TABLE_NAME,
            self.config_,
            Xs,
            self.graph_
        )

    def _more_tags(self) -> Dict[str, Any]:
        return {'no_validation': True}
