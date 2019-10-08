import json
import os
import pathlib

from typing import Any
from typing import Dict

import numpy as np

from automllib.utils import Timer
from model import Model


class AutoNLPDataset(object):
    def __init__(self, dataset_dir):
        self.dataset_name_ = dataset_dir
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self.read_metadata(os.path.join(dataset_dir, "meta.json"))

    def read_dataset(self):
        self.train_dataset = self._read_dataset(os.path.join(self.dataset_dir_, "train.data"))
        self.train_label = self.read_label(os.path.join(self.dataset_dir_, "train.solution"))
        self.test_dataset = self._read_dataset(os.path.join(self.dataset_dir_, "test.data"))

    def get_train(self):
        return self.train_dataset, self.train_label

    def get_test(self):
        return self.test_dataset

    def get_metadata(self):
        return self.metadata_

    def read_metadata(self, metadata_path):
        return json.load(open(metadata_path))

    def _read_dataset(self, dataset_path):
        with open(dataset_path, encoding='utf-8_sig') as fin:
            return fin.readlines()

    def read_label(self, label_path):
        return np.loadtxt(label_path)


def test_model() -> None:
    dataset = AutoNLPDataset('data/autonlp19/DEMO/DEMO.data')

    dataset.read_dataset()

    metadata = dataset.get_metadata()
    X_train, y_train = dataset.get_train()
    X_test = dataset.get_test()

    timer = Timer(metadata['time_budget'])

    timer.start()

    model = Model(metadata)

    while not model.done_training:
        model.train((X_train, y_train))

        probas = model.test(X_test)

        timer.check_remaining_time()
