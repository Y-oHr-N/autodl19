import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q jieba-fast')
os.system('pip3 install -q optuna')

import collections
import re
import time

from typing import Sequence

import jieba_fast as jieba
import lightgbm as lgb
import numpy as np

from imblearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from automllib.under_sampling import ModifiedRandomUnderSampler

CHINESE_STOP_WORDS = frozenset([
    'the', 'of', 'is', 'and',
    'to', 'in', 'that', 'we',
    'for', 'an', 'are', 'by',
    'be', 'as', 'on', 'with',
    'can', 'if', 'from', 'which',
    'you', 'it', 'this', 'then',
    'at', 'have', 'all', 'not',
    'one', 'has', 'or', 'that',
    '的', '了', '和', '是',
    '就', '都', '而', '及',
    '與', '著', '或', '一個',
    '沒有', '我們', '你們', '妳們',
    '他們', '她們', '是否'
])

REPLACE_BY_SPACE_RE_EN = re.compile('["/(){}\[\]\|@,;]')
REPLACE_BY_SPACE_RE_ZH = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
BAD_SYMBOLS_RE_EN = re.compile('[^0-9a-zA-Z #+_]')


def english_preprocessor(doc: str) -> str:
    doc = doc.lower()
    doc = REPLACE_BY_SPACE_RE_EN.sub(' ', doc)
    doc = BAD_SYMBOLS_RE_EN.sub('', doc)

    return doc.strip()


def chinese_preprocessor(doc: str) -> str:
    doc = REPLACE_BY_SPACE_RE_ZH.sub(' ', doc)

    return doc.strip()


def chinese_tokenizer(doc: str) -> Sequence[str]:
    return jieba.cut(doc, HMM=False)


class Model(object):
    def __init__(self, metadata):
        self.done_training = False
        self.metadata = metadata

    def train(self, train_dataset, remaining_time_budget=None):
        X_train, y_train = train_dataset
        y_train = np.argmax(y_train, axis=1)
        random_state=0

        if self.metadata['language'] == 'ZH':
            preprocessor = chinese_preprocessor
            stop_words = CHINESE_STOP_WORDS
            tokenizer = chinese_tokenizer
        else:
            preprocessor = english_preprocessor
            stop_words = 'english'
            tokenizer = None

        vectorizer = TfidfVectorizer(
            dtype=np.float32,
            max_features=10_000,
            max_df=0.95,
            min_df=2,
            preprocessor=preprocessor,
            stop_words=stop_words,
            tokenizer=tokenizer
        )
        reducer = TruncatedSVD(n_components=100, random_state=random_state)
        sampler = ModifiedRandomUnderSampler(random_state=random_state, verbose=1)
        model = lgb.LGBMClassifier(n_jobs=-1, random_state=random_state)

        self.model_ = make_pipeline(vectorizer, reducer, sampler, model)

        print(self.metadata)
        print(collections.Counter(y_train))

        self.model_.fit(X_train, y_train)

        self.done_training = True

    def test(self, X_test, remaining_time_budget=None):
        return self.model_.predict_proba(X_test)
