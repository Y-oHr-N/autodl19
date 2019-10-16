import logging
import os
import time

os.system('pip3 install -q kapre')

import keras
import numpy as np
import tensorflow as tf

from kapre.time_frequency import Melspectrogram
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import safe_indexing
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.data_utils import Sequence

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)
logger.setLevel(logging.INFO)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.compat.v1.Session(config=config)

set_session(sess)

SAMPLING_FREQ = 16_000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1_024  # 0.064 sec
FMIN = 20
FMAX = SAMPLING_FREQ // 2


def make_logmel_model(input_shape):
    model = keras.models.Sequential()

    model.add(
        Melspectrogram(
            fmax=FMAX,
            fmin=FMIN,
            n_dft=N_FFT,
            n_hop=HOP_LENGTH,
            n_mels=N_MELS,
            name='melgram',
            image_data_format='channels_last',
            input_shape=input_shape,
            return_decibel_melgram=True,
            power_melgram=2.0,
            sr=SAMPLING_FREQ,
        )
    )

    return model


def get_fixed_array(X_list, len_sample=5, sr=SAMPLING_FREQ):
    n = len(X_list)

    for i in range(n):
        if n < len_sample * sr:
            n_repeat = np.ceil(
                sr * len_sample / X_list[i].shape[0]
            ).astype(np.int32)
            X_list[i] = np.tile(X_list[i], n_repeat)

        X_list[i] = X_list[i][:len_sample * sr]

    X = np.asarray(X_list)
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)

    return X


def get_kapre_logmel(X_list, model, len_sample=5):
    X = get_fixed_array(X_list, len_sample=len_sample)
    X = model.predict(X)
    X = X.transpose(0, 2, 1, 3)

    return X


def make_cnn_model(input_shape, n_classes, max_layer_num=5):
    model = Sequential()
    min_size = min(input_shape[:2])
    optimizer = tf.keras.optimizers.SGD(decay=1e-06)

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
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(optimizer, 'categorical_crossentropy')

    return model


class CutOut(object):
    def __init__(self, probability=0.5, F=0.2):
        self.probability = probability
        self.F = F

    def __call__(self, image):
        _, w, _ = image.shape
        p = np.random.rand()

        if p > self.probability:
            return image

        f = np.random.randint(0, int(w * self.F))
        f0 = np.random.randint(0, w - f)

        image[:, f0:f0 + f, :] = 0

        return image


class RandomCropGenerator(Sequence):
    def __init__(self, X, y=None, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        n, _, _, _ = self.X.shape

        return int(np.ceil(n / self.batch_size))

    def __getitem__(self, i):
        batch = slice(i * self.batch_size, (i + 1) * self.batch_size)

        X = self.X[batch]
        n, h, w, _ = X.shape
        Xt = np.zeros((n, w, w, 1))

        for i in range(n):
            h0 = np.random.randint(0, h - w)
            Xt[i] = X[i, h0:h0 + w, :, :]

        y = self.y

        if y is not None:
            y = self.y[batch]

        return Xt, y


class MixupGenerator(object):
    def __init__(
        self,
        X,
        y,
        alpha=0.2,
        batch_size=32,
        datagen=None,
        shuffle=True
    ):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.batch_size = batch_size
        self.datagen = datagen
        self.shuffle = shuffle

    def __call__(self):
        while True:
            n, _, _, _ = self.X.shape
            indices = np.arange(n)

            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(int(n / 2 / self.batch_size)):
                # random crop
                datagen = RandomCropGenerator(
                    safe_indexing(self.X, indices),
                    safe_indexing(self.y, indices),
                    batch_size=self.batch_size
                )
                X1, y1 = datagen[2 * i]
                X2, y2 = datagen[2 * i + 1]

                # mixup
                l = np.random.beta(self.alpha, self.alpha, self.batch_size)
                X_l = l.reshape(self.batch_size, 1, 1, 1)
                y_l = l.reshape(self.batch_size, 1)
                X = X1 * X_l + X2 * (1.0 - X_l)
                y = y1 * y_l + y2 * (1.0 - y_l)

                if self.datagen is not None:
                    # cutout
                    for i in range(self.batch_size):
                        X[i] = self.datagen.random_transform(X[i])
                        X[i] = self.datagen.standardize(X[i])

                yield X, y


class Model(object):
    def __init__(
        self,
        metadata,
        batch_size=32,
        n_predictions=10,
        patience=100,
        random_state=0
    ):
        self.batch_size = batch_size
        self.metadata = metadata
        self.n_predictions = n_predictions
        self.patience = patience
        self.random_state = random_state

        self.done_training = False
        self.max_score = 0
        self.n_iter = 0

    def train(self, train_dataset, remaining_time_budget=None):
        start_time = time.perf_counter()

        if not hasattr(self, 'X_train'):
            self.logmel_model = make_logmel_model((1, 5 * SAMPLING_FREQ))

            X_train, y_train = train_dataset
            X_train = get_kapre_logmel(X_train, self.logmel_model)
            X_train = (
                X_train - np.mean(
                    X_train,
                    axis=(1, 2, 3),
                    keepdims=True
                )
            ) / np.std(X_train, axis=(1, 2, 3), keepdims=True)

            self.X_train, self.X_valid, \
                self.y_train, self.y_valid = train_test_split(
                    X_train,
                    y_train,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=y_train,
                    train_size=0.9
                )
            self.train_size, _, w, _ = self.X_train.shape

            logger.info(f'X.shape={X_train.shape}')

            self.model = make_cnn_model((w, w, 1), self.metadata['class_num'])


        while True:
            elapsed_time = time.perf_counter() - start_time
            remaining_time = remaining_time_budget - elapsed_time

            if remaining_time <= 0.125 * self.metadata['time_budget']:
                self.done_training = True

                break

            datagen = ImageDataGenerator(
                preprocessing_function=CutOut()
            )
            training_generator = MixupGenerator(
                self.X_train,
                self.y_train,
                batch_size=self.batch_size,
                datagen=datagen
            )()
            valid_generator = RandomCropGenerator(
                self.X_valid,
                batch_size=self.batch_size
            )

            self.model.fit_generator(
                training_generator,
                epochs=self.n_iter + 1,
                initial_epoch=self.n_iter,
                shuffle=True,
                steps_per_epoch=self.train_size // self.batch_size,
                verbose=1
            )

            probas = self.model.predict_generator(valid_generator)
            valid_score = roc_auc_score(self.y_valid, probas, average='macro')

            self.n_iter += 1

            logger.info(
                f'valid_auc={valid_score:.3f}, '
                f'max_valid_auc={self.max_score:.3f}'
            )

            if self.max_score < valid_score:
                self.max_score = valid_score

                break

    def test(self, X_test, remaining_time_budget=None):
        if not hasattr(self, 'X_test'):
            self.X_test = get_kapre_logmel(X_test, self.logmel_model)
            self.X_test = (
                self.X_test - np.mean(
                    self.X_test,
                    axis=(1, 2, 3),
                    keepdims=True
                )
            ) / np.std(self.X_test, axis=(1, 2, 3), keepdims=True)

        probas = np.zeros(
            (self.metadata['test_num'], self.metadata['class_num'])
        )

        for _ in range(self.n_predictions):
            test_generator = RandomCropGenerator(
                self.X_test,
                batch_size=self.batch_size
            )

            probas += self.model.predict_generator(test_generator)

        probas /= self.n_predictions

        return probas
