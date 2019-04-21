import datetime
import json
import logging
import os
import pathlib

from typing import Any
from typing import Dict
from typing import Union

import numpy as np
import pandas as pd

try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None

from sklearn.metrics import roc_auc_score

from automllib.constants import TYPE_MAP
from automllib.utils import Timer
from model import Model

logger = logging.getLogger(__name__)


def date_parser(x: Union[float, str]) -> datetime.datetime:
    x = float(x)

    if np.isnan(x):
        return np.nan

    return datetime.datetime.fromtimestamp(x / 1000.0)


def load_info(path: pathlib.Path) -> Dict[str, Any]:
    info_path = path / 'train' / 'info.json'

    with info_path.open() as f:
        info = json.load(f)

    return info


def load_train_data(path: pathlib.Path, info: Dict[str, Any]) -> pd.DataFrame:
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


def load_train_label(path: pathlib.Path) -> pd.Series:
    label_path = path / 'train' / 'main_train.solution.gz'

    return pd.read_csv(label_path, squeeze=True)


def load_test_data(path: pathlib.Path, info: Dict[str, Any]) -> pd.DataFrame:
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


def load_test_label(path: pathlib.Path) -> pd.Series:
    label_path = path / 'main_test.solution.gz'

    return pd.read_csv(label_path, squeeze=True)


def make_experiment() -> Experiment:
    try:
        api_key = os.getenv('COMET_API_KEY')

        experiment = Experiment(
            api_key=api_key,
            project_name='automl-kddcup19'
        )

        build_number = os.getenv('TRAVIS_BUILD_NUMBER')

        if build_number is not None:
            experiment.log_other('Build Number', build_number)

    except Exception:
        experiment = None

    return experiment


def test_model() -> None:
    data_path = pathlib.Path('data')
    ref_path = pathlib.Path('ref')
    probabilities = {}
    experiment = make_experiment()

    for path in data_path.iterdir():
        info = load_info(path)
        train_data = load_train_data(path, info)
        train_label = load_train_label(path)
        test_data = load_test_data(path, info)

        logger.info(f'Loaded data from {path.name}.')

        timer = Timer(info['time_budget'])
        model = Model(info)

        model.fit(train_data, train_label, timer.get_remaining_time())

        y_score = model.predict(test_data, timer.get_remaining_time())

        assert len(test_data) == len(y_score)
        assert y_score.isnull().sum() == 0

        probabilities[path.name] = y_score

        timer.check_remaining_time()

    for path in ref_path.iterdir():
        test_label = load_test_label(path)

        logger.info(f'Loaded ref from {path.name}.')

        score = roc_auc_score(test_label, probabilities[path.name])

        logger.info(f'The AUC is {score:.3f}.')

        assert score > 0.5

        if experiment is not None:
            experiment.log_metric(f'AUC of {path.name}', score)
