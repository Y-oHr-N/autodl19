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

from automllib.automl import AutoMLClassifier
from automllib.table_join import TYPE_MAP
from automllib.utils import Timer

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
    label_path = path / 'test' / 'main_test.solution.gz'

    return pd.read_csv(label_path, squeeze=True)


def make_experiment() -> Experiment:
    try:
        api_key = os.getenv('COMET_API_KEY')

        experiment = Experiment(
            api_key=api_key,
            project_name='automl-kddcup19'
        )

        build_number = os.getenv('TRAVIS_BUILD_NUMBER', default=False)
        pull_request = os.getenv('TRAVIS_PULL_REQUEST', default=False)

        if build_number:
            experiment.log_other('Build number', build_number)

        if pull_request:
            experiment.log_other('Pull request number', pull_request)

    except Exception as e:
        logger.exception(e)

        experiment = None

    return experiment


def test_automl_classifier() -> None:
    data_path = pathlib.Path('data')
    ref_path = pathlib.Path('ref')
    probabilities = {}
    scores = []
    experiment = make_experiment()

    for path in data_path.iterdir():
        info = load_info(path)
        info['n_jobs'] = -1
        info['n_seeds'] = 12
        info['random_state'] = 0
        info['verbose'] = 1
        related_tables = load_train_data(path, info)
        X_train = related_tables.pop('main')
        y_train = load_train_label(path)
        X_test = load_test_data(path, info)

        logger.info(f'Loaded data from {path.name}.')

        timer = Timer(info['time_budget'])

        timer.start()

        model = AutoMLClassifier(**info)

        model.fit(X_train, y_train, related_tables=related_tables)

        if experiment is not None:
            experiment.log_parameters(model.best_params_)

        probas = model.predict_proba(X_test)

        assert len(X_test) == len(probas)

        probabilities[path.name] = probas[:, 1]

        timer.check_remaining_time()

    for path in data_path.iterdir():
        y_test = load_test_label(path)

        logger.info(f'Loaded ref from {path.name}.')

        score = roc_auc_score(y_test, probabilities[path.name])

        logger.info(f'The AUC of {path.name} is {score:.3f}.')

        assert score > 0.5

        scores.append(score)

        if experiment is not None:
            experiment.log_metric(f'AUC of {path.name}', score)

    with np.errstate(invalid='ignore'):
        mean_score = np.mean(scores)

    logger.info(f'The mean AUC is {mean_score:.3f}.')
