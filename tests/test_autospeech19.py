import json
import logging
import os
import pathlib
import pickle

from glob import glob
from typing import Any
from typing import Dict

import numpy as np

from sklearn.metrics import roc_auc_score

from automllib.utils import Timer
from model import Model

logger = logging.getLogger(__name__)


def get_solution(solution_dir):
  solution_names = sorted(ls(os.path.join(solution_dir, '*.solution')))
  solution_file = solution_names[0]

  return read_array(solution_file)


def ls(filename):
    return sorted(glob(filename))


def read_array(filename):
    array = np.loadtxt(filename)

    if len(array.shape) == 1:
        array = array.reshape(-1, 1)

    return array


class AutoSpeechDataset(object):
    def __init__(self, dataset_dir):
        self.dataset_name_ = dataset_dir
        self.dataset_dir_ = dataset_dir
        self.metadata_ = self.read_metadata(os.path.join(dataset_dir, "meta.json"))

    def read_dataset(self):
        self.train_dataset = self._read_dataset(os.path.join(self.dataset_dir_, "train.pkl"))
        self.train_label = self.read_label(os.path.join(self.dataset_dir_, "train.solution"))
        self.test_dataset = self._read_dataset(os.path.join(self.dataset_dir_, "test.pkl"))

    def get_train(self):
        return self.train_dataset, self.train_label

    def get_test(self):
        return self.test_dataset

    def get_metadata(self):
        return self.metadata_

    def read_metadata(self, metadata_path):
        return json.load(open(metadata_path))

    def _read_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as fin:
            return pickle.load(fin)

    def read_label(self, label_path):
        return np.loadtxt(label_path)


def test_model() -> None:
    dataset = AutoSpeechDataset('data/autospeech19/DEMO/DEMO.data')

    dataset.read_dataset()

    metadata = dataset.get_metadata()
    X_train, y_train = dataset.get_train()
    X_test = dataset.get_test()
    y_test = get_solution('data/autospeech19/DEMO')

    timer = Timer(metadata['time_budget'])

    timer.start()

    model = Model(metadata)

    while not model.done_training:
        remaining_time = timer.get_remaining_time()

        model.train((X_train, y_train), remaining_time_budget=remaining_time)

        probas = model.test(X_test)

        timer.check_remaining_time()

        assert probas.shape == y_test.shape

        score = roc_auc_score(y_test, probas, average='macro')

        assert score >= 0.5

        logger.info(f'test_auc={score:.3f}')
