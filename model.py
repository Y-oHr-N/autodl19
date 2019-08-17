import re

import lightgbm as lgb
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from automllib.ensemble import LGBMClassifierCV

MAX_VOCAB_SIZE = 10000


def clean_en_text(text):
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
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
    return ' '.join(jieba.cut(text, cut_all=False))


class Model(object):
    def __init__(self, metadata):
        self.done_training = False
        self.metadata = metadata

    def train(self, train_dataset, remaining_time_budget=None):
        if self.metadata['language'] == 'ZH':
            preprocessor = clean_zh_text
            tokenizer = _tokenize_chinese_words
        else:
            preprocessor = clean_en_text
            tokenizer = None

        self.vectorizer_ = TfidfVectorizer(
            max_features=MAX_VOCAB_SIZE,
            preprocessor=preprocessor,
            tokenizer=tokenizer
        )
        self.model_ = LGBMClassifierCV(
            n_jobs=-1,
            n_seeds=4,
            random_state=0
        )

        x_train, y_train = train_dataset
        x_train = self.vectorizer_.fit_transform(x_train)
        y_train = np.argmax(y_train, axis=1)

        print(f'{x_train.shape}')
        print(f'{len(np.unique(y_train))}')

        self.model_.fit(x_train, y_train)

        self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        x_test = self.vectorizer_.transform(x_test)

        return self.model_.predict_proba(x_test)
