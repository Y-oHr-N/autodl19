import glob
import json
import logging
import os

from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from automllib.utils import Timer
from model import Model

TYPE_MAP = {
    'time': str,
    'cat': str,
    'multi-cat': str,
    'num': np.float64
}

logger = logging.getLogger(__name__)


def ls(filename):
    return sorted(glob.glob(filename))


def get_solution(solution_dir):
    solution_names = sorted(ls(os.path.join(solution_dir, '*.solution.gz')))

    solution_file = solution_names[0]
    solution = pd.read_csv(solution_file)

    return solution


class AutoWSLDataset:
    def __init__(self, dataset_dir):
        self.dataset_name_ = dataset_dir
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self._read_metadata(
            os.path.join(dataset_dir, "info.json"))
        self.train_dataset = None
        self.train_label = None
        self.test_dataset = None

    def read_dataset(self):
        self.train_dataset = self._read_dataset(
            os.path.join(self.dataset_dir_, "train.data.gz"))
        self.train_label = self.read_label(
            os.path.join(self.dataset_dir_, "train.solution.gz"))
        self.test_dataset = self._read_dataset(
            os.path.join(self.dataset_dir_, "test.data.gz"))

    def get_train(self):
        if self.train_dataset is None:
            self.train_dataset = self._read_dataset(
                os.path.join(self.dataset_dir_, "train.data.gz"))
            self.train_label = self.read_label(
                os.path.join(self.dataset_dir_, "train.solution.gz"))

        return self.train_dataset, self.train_label

    def get_test(self):
        if self.test_dataset is None:
            self.test_dataset = self._read_dataset(
                os.path.join(self.dataset_dir_, "test.data.gz"))

        return self.test_dataset

    def get_metadata(self):
        return self.metadata_

    @staticmethod
    def _read_metadata(metadata_path):
        return json.load(open(metadata_path))

    def _read_dataset(self, dataset_path):
        schema = self.metadata_['schema']
        table_dtype = {key: TYPE_MAP[val] for key, val in schema.items()}
        date_list = [key for key, val in schema.items() if val == 'time']
        date_parser = (
            lambda millisecs: millisecs if np.isnan(float(millisecs))
            else datetime.fromtimestamp(float(millisecs)/1000)
        )

        dataset = pd.read_csv(
            dataset_path, sep='\t', dtype=table_dtype,
            parse_dates=date_list, date_parser=date_parser)

        return dataset

    @staticmethod
    def read_label(label_path):
        train_label = pd.read_csv(label_path)['label']
        return train_label


def test_model() -> None:
    dataset = AutoWSLDataset('data/autowsl19/DEMO/data')

    dataset.read_dataset()

    metadata = dataset.get_metadata()
    X_train, y_train = dataset.get_train()
    X_test = dataset.get_test()
    y_test = get_solution('data/autowsl19/DEMO/solution')

    timer = Timer(metadata['time_budget'])

    timer.start()

    model = Model(metadata)

    model.train(X_train, y_train)

    timer.check_remaining_time()

    timer = Timer(metadata['pred_time_budget'])

    timer.start()

    y_score = model.predict(X_test)

    timer.check_remaining_time()

    score = roc_auc_score(y_test, y_score)

    assert score >= 0.5

    logger.info(f'test_auc={score:.3f}')
