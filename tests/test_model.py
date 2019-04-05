from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
import pytest

from model import Model

ROOT_DIR = os.getcwd()
INPUT_DIR = os.path.join(ROOT_DIR, 'data')
TYPE_MAP = {'time': str, 'cat': str, 'multi-cat': str, 'num': float}


def read_info(datapath):
    with open(os.path.join(datapath, 'train', 'info.json'), 'r') as info_fp:
        info = json.load(info_fp)

    return info


def read_train(datapath, info):
    train_data = {}

    for table_name, columns in info['tables'].items():
        table_dtype = {key: TYPE_MAP[val] for key, val in columns.items()}

        if table_name == 'main':
            table_path = os.path.join(datapath, 'train', 'main_train.data.gz')
        else:
            table_path = os.path.join(datapath, 'train', f'{table_name}.data.gz')

        date_list = [key for key, val in columns.items() if val == 'time']

        train_data[table_name] = pd.read_csv(
            table_path,
            sep='\t',
            dtype=table_dtype,
            parse_dates=date_list,
            date_parser=lambda millisecs: millisecs \
                if np.isnan(float(millisecs)) \
                else datetime.fromtimestamp(float(millisecs)/1000)
        )

    train_label = pd.read_csv(
        os.path.join(datapath, 'train', 'main_train.solution.gz')
    )['label']

    return train_data, train_label


def read_test(datapath, info):
    main_columns = info['tables']['main']
    table_dtype = {key: TYPE_MAP[val] for key, val in main_columns.items()}
    table_path = os.path.join(datapath, 'test', 'main_test.data.gz')
    date_list = [key for key, val in main_columns.items() if val == 'time']

    test_data = pd.read_csv(
        table_path,
        sep='\t',
        dtype=table_dtype,
        parse_dates=date_list,
        date_parser=lambda millisecs: millisecs \
            if np.isnan(float(millisecs)) \
            else datetime.fromtimestamp(float(millisecs) / 1000)
    )

    return test_data


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_model() -> None:
    datanames = sorted(os.listdir(INPUT_DIR))

    for dataname in datanames:
        datapath = os.path.join(INPUT_DIR, dataname)
        info = read_info(datapath)
        train_data, train_label = read_train(datapath, info)
        test_data = read_test(datapath, info)

        model = Model(info)

        model.fit(train_data, train_label, np.inf)
        model.predict(test_data, np.inf)
