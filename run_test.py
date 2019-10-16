import json
import os
import pathlib
import pickle

from typing import Any
from typing import Dict

import numpy as np

from automllib.utils import Timer
from model import Model


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
    #dataset = AutoSpeechDataset('../../data/data01.data')

    dataset.read_dataset()

    metadata = dataset.get_metadata()
    X_train, y_train = dataset.get_train()
    X_test = dataset.get_test()

    timer = Timer(metadata['time_budget'])

    timer.start()

    model = Model(metadata)

    while not model.done_training:
        remaining_time = timer.get_remaining_time()

        model.train((X_train, y_train), remaining_time_budget=remaining_time)

        probas = model.test(X_test)

        timer.check_remaining_time()

        assert probas.shape == (metadata['test_num'], metadata['class_num'])

if __name__ == '__main__':
    test_model()
