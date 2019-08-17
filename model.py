import os

os.system('pip3 install -q colorlog')
os.system('pip3 install -q imbalanced-learn')
os.system('pip3 install -q lightgbm')
os.system('pip3 install -q optuna')
os.system('pip3 install -q pandas==0.24.2')
os.system('pip3 install -q scikit-learn>=0.21.0')

import re

import jieba
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from automllib.ensemble import LGBMClassifierCV


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

    def train(self, train_dataset, remaining_time_budget=None):
        if self.metadata['language'] == 'ZH':
            preprocessor = clean_zh_text
            stop_words = None
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
        self.model_ = LGBMClassifierCV(
            n_estimators=1_000,
            n_iter_no_change=30,
            n_jobs=-1,
            n_seeds=4,
            random_state=0,
            verbose=1
        )

        x_train, y_train = train_dataset
        x_train = self.vectorizer_.fit_transform(x_train)
        x_train = self.reducer_.fit_transform(x_train)
        y_train = np.argmax(y_train, axis=1)

        print(self.metadata)

        self.model_.fit(x_train, y_train)

        self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        x_test = self.vectorizer_.transform(x_test)
        x_test = self.reducer_.transform(x_test)

        return self.model_.predict_proba(x_test)
