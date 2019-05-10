import logging

from collections import defaultdict
from collections import deque
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import pandas as pd

from .constants import AGGREGATE_FUNCTIONS_MAP as AFS_MAP
from .constants import CATEGORICAL_TYPE as C_TYPE
from .constants import MAIN_TABLE_NAME
from .constants import NUMERICAL_PREFIX
from .constants import NUMERICAL_TYPE as N_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import get_categorical_feature_names
from .utils import get_numerical_feature_names
from .utils import Timeit

logger = logging.getLogger(__name__)


def bfs(root_name, graph, tconfig):
    tconfig[MAIN_TABLE_NAME]['depth'] = 0
    queue = deque([root_name])

    while queue:
        u_name = queue.popleft()

        for edge in graph[u_name]:
            v_name = edge['to']

            if 'depth' not in tconfig[v_name]:
                tconfig[v_name]['depth'] = tconfig[u_name]['depth'] + 1
                queue.append(v_name)


@Timeit(logger)
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


@Timeit(logger)
def temporal_join(u, v, v_name, key, time_col):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    tmp_u.sort_values(by=[key, time_col], ascending=[True, True], inplace=True)

    tmp_u = tmp_u.groupby(key).ffill()

    # TODO: Check exceptions for all relations (one2one, one2many, many2many)
    tmp_u.columns = tmp_u.columns.map(
        lambda a: f"{a}_NEAREST({v_name})" if not (a == key or a == time_col) else a
    )
    tmp_u.drop([key, time_col], axis=1, inplace=True)  # Drop the key column.

    if tmp_u.empty:
        logger.info('Return u because temp_u is empty.')
        return u

    ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)
    del tmp_u
    return ret


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


@Timeit(logger)
def merge_table(tables, config):
    graph = defaultdict(list)

    for rel in config['relations']:
        ta = rel['table_A']
        tb = rel['table_B']

        graph[ta].append({
            "to": tb,
            "key": rel['key'],
            "type": rel['type']
        })

        graph[tb].append({
            "to": ta,
            "key": rel['key'],
            "type": '_'.join(rel['type'].split('_')[::-1])
        })

    bfs(MAIN_TABLE_NAME, graph, config['tables'])

    return dfs(MAIN_TABLE_NAME, config, tables, graph)


def aggregate_functions(
    X: TWO_DIM_ARRAY_TYPE
) -> Dict[str, List[Union[str, Callable]]]:
    func = {}

    c_feature_names = get_categorical_feature_names(X)
    # m_feature_names = get_multi_value_categorical_feature_names(X)
    n_feature_names = get_numerical_feature_names(X)
    # t_feature_names = get_time_feature_names(X)

    func.update({name: AFS_MAP[C_TYPE] for name in c_feature_names})
    # func.update({name: AFS_MAP[M_TYPE] for name in m_feature_names})
    func.update({name: AFS_MAP[N_TYPE] for name in n_feature_names})
    # func.update({name: AFS_MAP[T_TYPE] for name in t_feature_names})

    return func


class Config(object):
    def __init__(self, info):
        self.data = info.copy()
        self.data['tables'] = {}

        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data
