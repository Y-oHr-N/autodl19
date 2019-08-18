import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q optuna')

import collections
import re
import time

import jieba
import lightgbm as lgb
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from automllib.under_sampling import RandomUnderSampler

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


def clean_en_text(text):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.strip()

    return text


def clean_zh_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = text.strip()

    return text


def _tokenize_chinese_words(text):
    return list(jieba.cut(text, cut_all=False))


class Model(object):
    def __init__(self, metadata):
        self.done_training = False
        self.metadata = metadata
        self.epoch = 0

    def train(self, train_dataset, remaining_time_budget=None):
        X_train, y_train = train_dataset
        y_train = np.argmax(y_train, axis=1)

        if self.epoch == 0:
            self.model_ = DummyClassifier()

            self.model_.fit(X_train, y_train)

            self.epoch += 1

            return

        if self.metadata['language'] == 'ZH':
            preprocessor = clean_zh_text
            stop_words = CHINESE_STOP_WORDS
            tokenizer = _tokenize_chinese_words
        else:
            preprocessor = clean_en_text
            stop_words = 'english'
            tokenizer = None

        self.vectorizer_ = TfidfVectorizer(
            dtype='float32',
            max_features=10_000,
            max_df=0.95,
            min_df=2,
            preprocessor=preprocessor,
            stop_words=stop_words,
            tokenizer=tokenizer
        )
        self.reducer_ = TruncatedSVD(n_components=100, random_state=0)
        self.sampler_ = RandomUnderSampler(random_state=0, verbose=1)
        self.model_ = lgb.LGBMClassifier(n_jobs=-1, random_state=0)

        print(self.metadata)
        print(collections.Counter(y_train))

        start_time = time.perf_counter()
        X_train = self.vectorizer_.fit_transform(X_train)
        print(f'elapsed_time={time.perf_counter() - start_time:.3f}')

        start_time = time.perf_counter()
        X_train = self.reducer_.fit_transform(X_train)
        print(f'elapsed_time={time.perf_counter() - start_time:.3f}')

        X_train, y_train = self.sampler_.fit_resample(X_train, y_train)

        self.model_.fit(X_train, y_train)

        self.done_training = True

    def test(self, X_test, remaining_time_budget=None):
        if self.epoch > 0:
            X_test = self.vectorizer_.transform(X_test)
            X_test = self.reducer_.transform(X_test)

        return self.model_.predict_proba(X_test)
