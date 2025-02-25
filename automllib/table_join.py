import logging

from collections import defaultdict
from collections import deque
from typing import Any
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd

# from scipy.stats import kurtosis

from .base import BaseTransformer
from .base import ONE_DIM_ARRAYLIKE_TYPE
from .base import TWO_DIM_ARRAYLIKE_TYPE

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
    prefix: str,
    indices: bool = False
) -> ONE_DIM_ARRAYLIKE_TYPE:
    logger = logging.getLogger(__name__)
    is_startwith = X.columns.str.startswith(prefix)
    n_features = is_startwith.sum()

    logger.info(f'Number of features starting with {prefix} is {n_features}.')

    if indices:
        return np.where(is_startwith)[0]
    else:
        return X.columns[is_startwith]


def get_categorical_feature_names(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    method: str = 'prefix',
    indices: bool = False
) -> ONE_DIM_ARRAYLIKE_TYPE:
    if method == 'prefix':
        return get_feature_names_by_prefix(
            X,
            CATEGORICAL_PREFIX,
            indices=indices
        )

    elif method == 'dtype':
        X = X.select_dtypes('category')

        return X.columns

    else:
        raise ValueError(f'Invalid method: {method}.')


def get_multi_value_categorical_feature_names(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    method: str = 'prefix',
    indices: bool = False
) -> ONE_DIM_ARRAYLIKE_TYPE:
    if method == 'prefix':
        return get_feature_names_by_prefix(
            X,
            MULTI_VALUE_CATEGORICAL_PREFIX,
            indices=indices
        )

    elif method == 'dtype':
        X = X.select_dtypes('object')

        return X.columns

    else:
        raise ValueError(f'Invalid method: {method}.')


def get_numerical_feature_names(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    method: str = 'prefix',
    indices: bool = False
) -> ONE_DIM_ARRAYLIKE_TYPE:
    if method == 'prefix':
        return get_feature_names_by_prefix(
            X,
            NUMERICAL_PREFIX,
            indices=indices
        )

    elif method == 'dtype':
        X = X.select_dtypes('number')

        return X.columns

    else:
        raise ValueError(f'Invalid method: {method}.')


def get_time_feature_names(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    method: str = 'prefix',
    indices: bool = False
) -> ONE_DIM_ARRAYLIKE_TYPE:

    if method == 'prefix':
        return get_feature_names_by_prefix(X, TIME_PREFIX, indices=indices)

    elif method == 'dtype':
        X = X.select_dtypes('datetime')

        return X.columns

    else:
        raise ValueError(f'Invalid method: {method}.')


def join(u, v, u_name, v_name, key, type_, config):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    if type_.split("_")[2] == 'many':
        func = aggregate_functions(v.drop(columns=key))
        v = v.groupby(key).agg(func)
        v.columns = v.columns.map(
            lambda a: f"{NUMERICAL_PREFIX}{a[1].upper()}({a[0]})"
        )
    else:
        v = v.set_index(key)

    if type_.split("_")[0] == 'many':
        if u.columns.str.endswith(f'_BY({key})').sum() == 0:
            raw_columns = config[u_name]['type'].keys()
            func = aggregate_functions(u[raw_columns].drop(columns=key))
            intermediate = u.groupby(key).agg(func)
            intermediate.columns = intermediate.columns.map(
                lambda a: f"{NUMERICAL_PREFIX}{a[1].upper()}({a[0]})_BY({key})"
            )
            intermediate = intermediate.reset_index()
            u = pd.merge(u, intermediate, how='left', on=key)

    v.columns = v.columns.map(lambda a: f"{a.split('_', 1)[0]}_{v_name}.{a}")

    return u.join(v, on=key)


def temporal_join(u, v, u_name, v_name, key, type_, config, time_col):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    logger = logging.getLogger(__name__)
    tmp_u = u[[time_col, key]]
    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    tmp_u = tmp_u.sort_values(time_col)
    tmp_u = tmp_u.groupby(key).ffill()
    tmp_u.columns = tmp_u.columns.map(
        lambda a: f"{a}_NEAREST({v_name})" if not (a == key or a == time_col) else a
    )
    tmp_u = tmp_u.drop(columns=[key, time_col])

    if type_.split("_")[0] == 'many':
        if u.columns.str.endswith(f'_BY({key})').sum() == 0:
            raw_columns = config[u_name]['type'].keys()
            func = aggregate_functions(u[raw_columns].drop(columns=key))
            intermediate = u.groupby(key).agg(func)
            intermediate.columns = intermediate.columns.map(
                lambda a: f"{NUMERICAL_PREFIX}{a[1].upper()}({a[0]})_BY({key})"
            )
            intermediate = intermediate.reset_index()
            u = pd.merge(u, intermediate, how='left', on=key)
    if tmp_u.empty:
        logger.info('Return u because temp_u is empty.')

        return u

    return pd.concat([u, tmp_u.loc['u']], axis=1, join_axes=[u.index])


def dfs(u_name, config, Xs, graph, time_col):
    logger = logging.getLogger(__name__)
    u = Xs[u_name]

    logger.info(f'Enter {u_name}.')

    for edge in graph[u_name]:
        v_name = edge['to']

        if config[v_name]['depth'] <= config[u_name]['depth']:
            continue

        v = dfs(v_name, config, Xs, graph, time_col)
        key = edge['key']
        type_ = edge['type']

        if time_col not in u and time_col in v:
            continue

        if time_col in u and time_col in v:
            logger.info(f'Temporal Join {u_name} <--{type_}--t {v_name}.')

            u = temporal_join(u, v, u_name, v_name, key, type_, config, time_col)

        else:
            logger.info(f'Join {u_name} <--{type_}--nt {v_name}.')

            u = join(u, v, u_name, v_name, key, type_, config)

    logger.info(f'Leave {u_name}.')

    return u


def aggregate_functions(
    X: TWO_DIM_ARRAYLIKE_TYPE
) -> Dict[str, Sequence[Union[str, Callable]]]:
    AFS_MAP = {
        CATEGORICAL_TYPE: [
        #     'count',
            'nunique'
        ],
        MULTI_VALUE_CATEGORICAL_TYPE: [
        #     'count',
        #     'nunique'
        ],
        NUMERICAL_TYPE: [
            'count',
            'min',
            'max',
            'mean',
        #     'median',
        #     'sum',
        #     'std',
        #     'skew',
        #     kurtosis
        ],
        TIME_TYPE: [
        #     'count'
            lambda x: (x.max() - x.min()).total_seconds()
        ]
    }
    func = {}

    c_feature_names = get_categorical_feature_names(X)
    # m_feature_names = get_multi_value_categorical_feature_names(X)
    n_feature_names = get_numerical_feature_names(X)
    t_feature_names = get_time_feature_names(X)

    func.update({name: AFS_MAP[CATEGORICAL_TYPE] for name in c_feature_names})
    # func.update({name: AFS_MAP[MULTI_VALUE_CATEGORICAL_TYPE] for name in m_feature_names})
    func.update({name: AFS_MAP[NUMERICAL_TYPE] for name in n_feature_names})
    func.update({name: AFS_MAP[TIME_TYPE] for name in t_feature_names})

    return func


class TableJoiner(BaseTransformer):
    def __init__(
        self,
        main_table_name: str = 'main',
        relations: Sequence[Dict[str, str]] = None,
        tables: Dict[str, Dict[str, str]] = None,
        time_col: str = None,
        verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)

        self.main_table_name = main_table_name
        self.relations = relations
        self.tables = tables
        self.time_col = time_col

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE = None,
        related_tables: Dict[str, TWO_DIM_ARRAYLIKE_TYPE] = None,
    ) -> 'TableJoiner':
        self.graph_ = defaultdict(list)
        self.config_ = {}

        if related_tables is None:
            self.related_tables_ = {}
        else:
            self.related_tables_ = related_tables

        relations = self.relations or []
        tables = self.tables or {self.main_table_name: {}}

        for rel in relations:
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

        for tname, ttype in tables.items():
            self.config_[tname] = {}
            self.config_[tname]['type'] = ttype

        self.config_[self.main_table_name]['depth'] = 0

        queue = deque([self.main_table_name])

        while queue:
            u_name = queue.popleft()

            for edge in self.graph_[u_name]:
                v_name = edge['to']

                if 'depth' not in self.config_[v_name]:
                    self.config_[v_name]['depth'] = \
                        self.config_[u_name]['depth'] + 1

                    queue.append(v_name)

        return self

    def _transform(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        Xs = self.related_tables_.copy()
        Xs[self.main_table_name] = X

        return dfs(
            self.main_table_name,
            self.config_,
            Xs,
            self.graph_,
            self.time_col
        )

    def _more_tags(self) -> Dict[str, Any]:
        return {'no_validation': True}
