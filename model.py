import re

import jieba
import lightgbm as lgb
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

MAX_VOCAB_SIZE = 10000


def clean_en_text(dat):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    ret = []

    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()

        ret.append(line)

    return ret


def clean_zh_text(dat):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    ret = []

    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        ret.append(line)

    return ret


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


class Model(object):
    def __init__(self, metadata):
        self.done_training = False
        self.metadata = metadata

    def train(self, train_dataset, remaining_time_budget=None):
        self.vectorizer_ = TfidfVectorizer(max_features=MAX_VOCAB_SIZE)
        self.model_ = lgb.LGBMClassifier(random_state=0)

        x_train, y_train = train_dataset
        y_train = np.argmax(y_train, axis=1)

        if self.metadata['language'] == 'ZH':
            x_train = clean_zh_text(x_train)
            x_train = list(map(_tokenize_chinese_words, x_train))
        else:
            x_train = clean_en_text(x_train)

        x_train = self.vectorizer_.fit_transform(x_train)

        self.model_.fit(x_train, y_train)

        self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        if self.metadata['language'] == 'ZH':
            x_test = list(map(_tokenize_chinese_words, x_test))

        x_test = self.vectorizer_.transform(x_test)

        return self.model_.predict_proba(x_test)
