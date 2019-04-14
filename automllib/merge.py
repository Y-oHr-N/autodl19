from collections import defaultdict
from collections import deque
import logging

import pandas as pd

from .constants import MAIN_TABLE_NAME
from .constants import NUMERICAL_PREFIX
from .utils import aggregate_functions
from .utils import timeit

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


@timeit
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


@timeit
def temporal_join(u, v, v_name, key, time_col):
    if isinstance(key, list):
        assert len(key) == 1
        key = key[0]

    tmp_u = u[[time_col, key]]
    tmp_u = pd.concat([tmp_u, v], keys=['u', 'v'], sort=False)
    rehash_key = f'rehash_{key}'
    tmp_u[rehash_key] = tmp_u[key].apply(lambda x: hash(x) % 200)

    tmp_u.sort_values(time_col, inplace=True)

    columns = v.columns.drop(key)
    func = aggregate_functions(columns)
    tmp_u = tmp_u.groupby(rehash_key).rolling(5).agg(func)

    tmp_u.reset_index(0, drop=True, inplace=True)  # drop rehash index

    tmp_u.columns = tmp_u.columns.map(
        lambda a: f"{NUMERICAL_PREFIX}{a[1].upper()}_ROLLING5({v_name}.{a[0]})"
    )

    if tmp_u.empty:
        logger.info("empty tmp_u, return u")

        return u

    ret = pd.concat([u, tmp_u.loc['u']], axis=1, sort=False)

    del tmp_u

    return ret


def dfs(u_name, config, tables, graph):
    u = tables[u_name]

    logger.info(f"enter {u_name}")

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
            logger.info(f"join {u_name} <--{type_}--t {v_name}")
            u = temporal_join(u, v, v_name, key, config['time_col'])
        else:
            logger.info(f"join {u_name} <--{type_}--nt {v_name}")
            u = join(u, v, v_name, key, type_)

        del v

    logger.info(f"leave {u_name}")

    return u


@timeit
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
