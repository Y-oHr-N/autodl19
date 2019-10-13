import os
os.system('pip3 install -q dcase_util')

import dcase_util
import librosa
import logging
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import safe_indexing
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import utils

try:
    config = tf.ConfigProto()
except AttributeError:
    config = tf.compat.v1.ConfigProto()

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)
logger.setLevel(logging.INFO)

config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.Session(config=config)

set_session(sess)


def extract_mfcc(data, n_mfcc=24, sr=16_000):
    results = []

    for d in data:
        r = librosa.feature.mfcc(d, n_mfcc=n_mfcc, sr=sr)
        r = r.transpose()

        results.append(r)

    return results


def extract_logmel(data, sr=16_000):
    results = []

    mel_extractor = dcase_util.features.MelExtractor(
        n_mels=64,
        win_length_samples=2048,
        hop_length_samples=512,
        # win_length_seconds=0.04,
        # hop_length_seconds=0.02,
        fs=sr
    )

    for d in data:
        r = mel_extractor(d)  # n_bin x len
        r = r.transpose()  # len x n_bin

        results.append(r)

    return results


def pad_seq(data, pad_len):
    return pad_sequences(
        data,
        maxlen=pad_len,
        dtype='float32',
        padding='post'
    )


def cnn_model(input_shape, num_class, max_layer_num=5):
    model = Sequential()
    min_size = min(input_shape[:2])

    for i in range(max_layer_num):
        if i == 0:
            model.add(Conv2D(64, 3, input_shape=input_shape, padding='same'))
        else:
            model.add(Conv2D(64, 3, padding='same'))

        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        min_size //= 2

        if min_size < 2:
            break

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(rate=0.5))
    model.add(Activation('relu'))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    return model


def get_frequency_masking(p=0.5, F=0.2):
    def frequency_masking(input_img):
        _, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        # frequency masking
        f = np.random.randint(0, int(img_w * F))
        f0 = np.random.randint(0, img_w - f)
        c = input_img.mean()
        input_img[:, f0:f0 + f, :] = c

        return input_img

    return frequency_masking


class MixupGenerator(object):
    def __init__(
        self,
        X_train,
        y_train,
        # sample_weight=None,
        alpha=0.2,
        batch_size=32,
        datagen=None,
        shuffle=True
    ):
        self.X_train = X_train
        self.y_train = y_train
        # self.sample_weight = sample_weight
        self.alpha = alpha
        self.batch_size = batch_size
        self.datagen = datagen
        self.shuffle = shuffle

        self.sample_num = len(X_train)

    def __call__(self):
        while True:
            indices = self.__get_exploration_order()
            itr_num = int(self.sample_num // (2 * self.batch_size))

            for i in range(itr_num):
                indices_head = indices[
                    2 * i * self.batch_size:(2 * i + 1) * self.batch_size
                ]
                indices_tail = indices[
                    (2 * i + 1) * self.batch_size:(2 * i + 2) * self.batch_size
                ]

                yield self.__data_generation(indices_head, indices_tail)

    def __get_exploration_order(self):
        indices = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indices)

        return indices

    def __data_generation(self, indices_head, indices_tail):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = safe_indexing(self.X_train, indices_head)
        X2 = safe_indexing(self.X_train, indices_tail)
        X = X1 * X_l + X2 * (1.0 - X_l)

        y1 = safe_indexing(self.y_train, indices_head)
        y2 = safe_indexing(self.y_train, indices_tail)
        y = y1 * y_l + y2 * (1.0 - y_l)

        # sample_weight1 = safe_indexing(self.sample_weight, indices_head)
        # sample_weight2 = safe_indexing(self.sample_weight, indices_tail)
        # sample_weight = sample_weight1 * l + sample_weight2 * (1.0 - l)

        if self.datagen is not None:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        # return X, y, sample_weight
        return X, y


class Model(object):
    def __init__(self, metadata, batch_size=32, patience=100, random_state=0):
        self.metadata = metadata
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

        self.done_training = False
        self.max_auc = 0
        self.n_iter = 0
        self.not_improve_learning_iter = 0
        self.val_res = None

    def train(self, train_dataset, remaining_time_budget=None):
        if remaining_time_budget <= 0.125 * self.metadata['time_budget']:
            self.done_training = True

            return

        if not hasattr(self, 'train_x'):
            train_x, train_y = train_dataset
            # fea_x = extract_mfcc(train_x)
            fea_x, train_y = utils.make_cropped_dataset_5sec(train_x, train_y)

            self.max_len = max([len(_) for _ in fea_x])

            fea_x = pad_seq(fea_x, self.max_len)
            train_x = fea_x[:, :, :, np.newaxis]
            # sample_weight = compute_sample_weight('balanced', train_y)

            logger.info(f'X.shape={train_x.shape}')

            self.train_x, self.val_x, \
                self.train_y, self.val_y, = train_test_split(
                # self.sample_weight, _ = train_test_split(
                    train_x,
                    train_y,
                    # sample_weight,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=train_y,
                    train_size=0.9
                )

            num_class = self.metadata['class_num']

            self.model = cnn_model(self.train_x.shape[1:], num_class)

            optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-06)

            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

        datagen = ImageDataGenerator(
            preprocessing_function=get_frequency_masking()
        )
        training_generator = MixupGenerator(
            self.train_x,
            self.train_y,
            # self.sample_weight,
            alpha=0.2,
            batch_size=self.batch_size,
            datagen=datagen
        )()

        self.model.fit_generator(
            training_generator,
            steps_per_epoch=self.train_x.shape[0] // self.batch_size,
            epochs=self.n_iter + 1,
            initial_epoch=self.n_iter,
            shuffle=True,
            verbose=1
        )

        self.n_iter += 1

        self.val_res = self.model.predict_proba(self.val_x)

        val_auc = roc_auc_score(self.val_y, self.val_res, average='macro')

        logger.info(f'val_auc={val_auc:.3f}, max_auc={self.max_auc:.3f}')

        if self.max_auc < val_auc:
            self.not_improve_learning_iter = 0
            self.max_auc = val_auc
        else:
            self.not_improve_learning_iter += 1

        if self.not_improve_learning_iter >= self.patience:
            self.done_training = True

    def test(self, test_x, remaining_time_budget=None):
        if not hasattr(self, 'test_x'):
            # fea_x = extract_mfcc(test_x)
            fea_x, _ = utils.make_cropped_dataset_5sec(test_x)
            fea_x = pad_seq(fea_x, self.max_len)

            self.test_x = fea_x[:, :, :, np.newaxis]

        if self.not_improve_learning_iter == 0:
            self.test_res = self.model.predict_proba(self.test_x)

        return self.test_res
