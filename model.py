import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q jieba-fast')
os.system('pip3 install -q optuna')

import collections
import re

from typing import Sequence

import jieba_fast as jieba
import lightgbm as lgb
import numpy as np

from imblearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from automllib.under_sampling import ModifiedRandomUnderSampler

HALF_WIDTH_CHARS = ''.join(chr(0x21 + i) for i in range(94))
FULL_WIDTH_CHARS = ''.join(chr(0xff01 + i) for i in range(94))
F2H = str.maketrans(FULL_WIDTH_CHARS, HALF_WIDTH_CHARS)

ANY_EMAIL_ADDRESS = re.compile(r'[\w.+-]+@[\w-]+(\.[\w-]+)+')
ANY_URL = re.compile(r'http(s)?[^\s]+[\w]')
ANY_NUMBER = re.compile(r'[+-]?\d+(\.\d+)?')
ANY_SYMBOLS = re.compile(r'[!-/:-@[-`{-~]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-zA-Z #+_]')

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


def english_preprocessor(doc: str) -> str:
    doc = doc.translate(F2H)
    doc = doc.lower()
    doc = ANY_EMAIL_ADDRESS.sub('EMAILADDRESS', doc)
    doc = ANY_URL.sub('URL', doc)
    doc = ANY_NUMBER.sub('NUMBER', doc)
    doc = ANY_SYMBOLS.sub(' ', doc)
    doc = BAD_SYMBOLS_RE.sub('', doc)

    return doc.strip()


def chinese_preprocessor(doc: str) -> str:
    doc = doc.translate(F2H)
    doc = doc.lower()
    doc = ANY_EMAIL_ADDRESS.sub('EMAILADDRESS', doc)
    doc = ANY_URL.sub('URL', doc)
    doc = ANY_NUMBER.sub('NUMBER', doc)
    doc = ANY_SYMBOLS.sub(' ', doc)

    return doc.strip()


def chinese_tokenizer(doc: str) -> Sequence[str]:
    return jieba.cut(doc, HMM=False)


class Model(object):
    def __init__(self, metadata, random_state=0):
        self.done_training = False
        self.metadata = metadata
        self.random_state = random_state

    def train(self, train_dataset, remaining_time_budget=None):
        if self.metadata['language'] == 'ZH':
            preprocessor = chinese_preprocessor
            stop_words = CHINESE_STOP_WORDS
            tokenizer = chinese_tokenizer
        else:
            preprocessor = english_preprocessor
            stop_words = 'english'
            tokenizer = None

        self.vectorizer_ = TfidfVectorizer(
            dtype=np.float32,
            max_features=10_000,
            max_df=0.95,
            min_df=2,
            preprocessor=preprocessor,
            stop_words=stop_words,
            tokenizer=tokenizer
        )
        self.reducer_ = TruncatedSVD(
            n_components=100,
            random_state=self.random_state
        )
        self.sampler_ = ModifiedRandomUnderSampler(
            random_state=self.random_state
        )
        self.model_ = lgb.LGBMClassifier(
            n_estimators=1_000,
            n_jobs=-1,
            random_state=self.random_state
        )

        X, y = train_dataset
        X = self.vectorizer_.fit_transform(X)
        X = self.reducer_.fit_transform(X)
        y = np.argmax(y, axis=1)
        X, y = self.sampler_.fit_resample(X, y)
        X, X_valid, y, y_valid = train_test_split(
            X,
            y,
            random_state=self.random_state
        )

        self.model_.fit(
            X,
            y,
            early_stopping_rounds=30,
            eval_set=[(X_valid, y_valid)]
        )

        print(f'Best iteration: {self.model_.best_iteration_}.')
        print(f'Best score: {self.model_.best_score_}.')
        print(self.model_.score(X_valid, y_valid))

        self.done_training = True

    def test(self, X, remaining_time_budget=None):
        X = self.vectorizer_.transform(X)
        X = self.reducer_.transform(X)

        return self.model_.predict_proba(X)
