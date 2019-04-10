from datetime import datetime
import json
from pathlib import Path
from typing import Any
from typing import Dict

import pandas as pd
from sklearn.metrics import roc_auc_score


from model import Model
from package.constants import TYPE_MAP
from package.utils import Timer


def date_parser(x: str) -> datetime:
    x = float(x)

    if pd.isna(x):
        return x

    return datetime.fromtimestamp(x / 1000.0)


def read_info(path: Path) -> Dict[str, Any]:
    info_path = path / 'train' / 'info.json'

    with info_path.open() as f:
        info = json.load(f)

    return info


def read_train_data(path: Path, info: Dict[str, Any]) -> pd.DataFrame:
    train_data = {}

    for table_name, column_types in info['tables'].items():
        if table_name == 'main':
            table_path = path / 'train' / 'main_train.data.gz'
        else:
            table_path = path / 'train' / f'{table_name}.data.gz'

        dtype = {}
        parse_dates = []

        for column_name, column_type in column_types.items():
            dtype[column_name] = TYPE_MAP[column_type]

            if column_type == 'time':
                parse_dates.append(column_name)

        train_data[table_name] = pd.read_csv(
            table_path,
            date_parser=date_parser,
            dtype=dtype,
            parse_dates=parse_dates,
            sep='\t'
        )

    return train_data


def read_train_label(path: Path) -> pd.Series:
    label_path = path / 'train' / 'main_train.solution.gz'

    return pd.read_csv(label_path, squeeze=True)


def read_test_data(path: Path, info: Dict[str, Any]) -> pd.DataFrame:
    column_types = info['tables']['main']
    table_path = path / 'test' / 'main_test.data.gz'
    dtype = {}
    parse_dates = []

    for column_name, column_type in column_types.items():
        dtype[column_name] = TYPE_MAP[column_type]

        if column_type == 'time':
            parse_dates.append(column_name)

    return pd.read_csv(
        table_path,
        dtype=dtype,
        date_parser=date_parser,
        parse_dates=parse_dates,
        sep='\t'
    )


def read_test_label(path: Path) -> pd.Series:
    label_path = path / 'main_test.solution.gz'

    return pd.read_csv(label_path, squeeze=True)


def test_model() -> None:
    data_path = Path('data')
    ref_path = Path('ref')
    probabilities = {}

    for path in data_path.iterdir():
        info = read_info(path)
        train_data = read_train_data(path, info)
        train_label = read_train_label(path)
        test_data = read_test_data(path, info)

        timer = Timer(info['time_budget'])
        model = Model(info)

        model.fit(train_data, train_label, timer.remaining_time())

        probabilities[path.name] = model.predict(
            test_data,
            timer.remaining_time()
        )

        timer.remaining_time()

    for path in ref_path.iterdir():
        test_label = read_test_label(path)

        roc_auc_score(test_label, probabilities[path.name])
