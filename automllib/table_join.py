import logging

from collections import defaultdict
from collections import deque
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import pandas as pd

from .base import BaseTransformer
from .constants import AGGREGATE_FUNCTIONS_MAP as AFS_MAP
from .constants import CATEGORICAL_TYPE as C_TYPE
from .constants import MAIN_TABLE_NAME
from .constants import NUMERICAL_PREFIX
from .constants import NUMERICAL_TYPE as N_TYPE
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import get_categorical_feature_names
from .utils import get_numerical_feature_names
from .utils import Timeit

logger = logging.getLogger(__name__)


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
    tmp_u = tmp_u.sort_values(time_col)
    tmp_u = tmp_u.groupby(key).ffill()

    # TODO: Check exceptions for all relations (one2one, one2many, many2many)
    tmp_u.columns = tmp_u.columns.map(
        lambda a: f"{a}_NEAREST({v_name})" if not (a == key or a == time_col) else a
    )
    tmp_u = tmp_u.drop(columns=[key, time_col])

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


class TableJoiner(BaseTransformer):
    _attributes = ['config_', 'graph_']

    def __init__(
        self,
        info: Dict[str, Any],
        related_tables: Dict[str, TWO_DIM_ARRAY_TYPE],
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)

        self.info = info
        self.related_tables = related_tables

    def _check_params(self) -> None:
        pass

    def _fit(
        self,
        X: TWO_DIM_ARRAY_TYPE,
        y: ONE_DIM_ARRAY_TYPE = None
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

    def _transform(self, X: TWO_DIM_ARRAY_TYPE) -> TWO_DIM_ARRAY_TYPE:
        Xs = self.related_tables.copy()
        Xs[MAIN_TABLE_NAME] = X

        return dfs(
            MAIN_TABLE_NAME,
            self.config_,
            Xs,
            self.graph_
        )
