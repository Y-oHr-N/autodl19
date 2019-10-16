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
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()

logger.addHandler(handler)
logger.setLevel(logging.INFO)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True
config.log_device_placement = False

sess = tf.compat.v1.Session(config=config)

set_session(sess)

# parameters
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
            trainable_kernel=False
        )
    )

    return model


def get_fixed_array(X_list, len_sample=5):
    for i in range(len(X_list)):
        if len(X_list[i]) < len_sample * SAMPLING_FREQ:
            n_repeat = np.ceil(
                SAMPLING_FREQ * len_sample / X_list[i].shape[0]
            ).astype(np.int32)
            X_list[i] = np.tile(X_list[i], n_repeat)

        X_list[i] = X_list[i][:len_sample * SAMPLING_FREQ]

    X = np.asarray(X_list)
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)

    return X


def get_kapre_logmel(X_list, len_sample=5, model=None):
    X = get_fixed_array(X_list, len_sample=len_sample)
    X = model.predict(X)
    X = X.transpose(0, 2, 1, 3)

    return X


def get_crop_image(image):
    time_dim, base_dim, _ = image.shape
    crop = np.random.randint(0, time_dim - base_dim)
    image = image[crop:crop + base_dim, :, :]

    return image


def make_cnn_model(input_shape, n_classes, max_layer_num=5):
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
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model


def get_frequency_masking(p=0.5, F=0.2):
    def frequency_masking(input_img):
        _, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        f = np.random.randint(0, int(img_w * F))
        f0 = np.random.randint(0, img_w - f)

        input_img[:, f0:f0 + f, :] = 0

        return input_img

    return frequency_masking


class TTAGenerator(object):
    def __init__(self, X_test, batch_size):
        self.X_test = X_test
        self.batch_size = batch_size

        self.n_samples = len(X_test)

    def __call__(self):
        while True:
            for start in range(0, self.n_samples, self.batch_size):
                end = min(start + self.batch_size, self.n_samples)
                X_test_batch = self.X_test[start:end]

                yield self.__data_generation(X_test_batch)

    def __data_generation(self, X_test_batch):
        d, _, w, _ = X_test_batch.shape
        X = np.zeros((d, w, w, 1))

        for i in range(d):
            X[i] = get_crop_image(X_test_batch[i])

        return X, None


class MixupGenerator(object):
    def __init__(
        self,
        X_train,
        y_train,
        alpha=0.2,
        batch_size=32,
        datagen=None,
        shuffle=True
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.batch_size = batch_size
        self.datagen = datagen
        self.shuffle = shuffle

    def __call__(self):
        while True:
            indices = self.__get_exploration_order()
            n_samples = len(self.X_train)
            itr_num = int(n_samples // (2 * self.batch_size))

            for i in range(itr_num):
                indices_head = indices[
                    2 * i * self.batch_size:(2 * i + 1) * self.batch_size
                ]
                indices_tail = indices[
                    (2 * i + 1) * self.batch_size:(2 * i + 2) * self.batch_size
                ]

                yield self.__data_generation(indices_head, indices_tail)

    def __get_exploration_order(self):
        n_samples = len(self.X_train)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        return indices

    def __data_generation(self, indices_head, indices_tail):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1_tmp = safe_indexing(self.X_train, indices_head)
        X2_tmp = safe_indexing(self.X_train, indices_tail)
        d, _, w, _ = X1_tmp.shape
        X1 = np.zeros((d, w, w, 1))
        X2 = np.zeros((d, w, w, 1))

        for i in range(self.batch_size):
            X1[i] = get_crop_image(X1_tmp[i])
            X2[i] = get_crop_image(X2_tmp[i])

        X = X1 * X_l + X2 * (1.0 - X_l)

        y1 = safe_indexing(self.y_train, indices_head)
        y2 = safe_indexing(self.y_train, indices_tail)
        y = y1 * y_l + y2 * (1.0 - y_l)

        if self.datagen is not None:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        return X, y


class Model(object):
    def __init__(
        self,
        metadata,
        batch_size=32,
        n_predictions=10,
        patience=100,
        random_state=0
    ):
        self.metadata = metadata
        self.n_predictions = n_predictions
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

        self.done_training = False
        self.max_score = 0
        self.n_iter = 0

    def train(self, train_dataset, remaining_time_budget=None):
        start_time = time.perf_counter()

        if not hasattr(self, 'train_x'):
            self.logmel_model = make_logmel_model((1, 5 * SAMPLING_FREQ))

            train_x, train_y = train_dataset
            train_x = get_kapre_logmel(
                train_x,
                len_sample=5,
                model=self.logmel_model
            )
            train_x = (
                train_x - np.mean(
                    train_x,
                    axis=(1, 2, 3),
                    keepdims=True
                )
            ) / np.std(train_x, axis=(1, 2, 3), keepdims=True)

            self.train_x, self.valid_x, \
                self.train_y, self.valid_y = train_test_split(
                    train_x,
                    train_y,
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=train_y,
                    train_size=0.9
                )
            self.train_size, _, w, _ = self.train_x.shape
            self.valid_size, _, _, _ = self.valid_x.shape

            logger.info(f'X.shape={train_x.shape}')

            self.model = make_cnn_model((w, w, 1), self.metadata['class_num'])

            optimizer = tf.keras.optimizers.SGD(decay=1e-06)

            self.model.compile(optimizer, 'categorical_crossentropy')

        while True:
            elapsed_time = time.perf_counter() - start_time
            remaining_time = remaining_time_budget - elapsed_time

            if remaining_time <= 0.125 * self.metadata['time_budget']:
                self.done_training = True

                break

            datagen = ImageDataGenerator(
                preprocessing_function=get_frequency_masking()
            )
            training_generator = MixupGenerator(
                self.train_x,
                self.train_y,
                batch_size=self.batch_size,
                datagen=datagen
            )()
            valid_generator = TTAGenerator(
                self.valid_x,
                batch_size=self.batch_size
            )()

            self.model.fit_generator(
                training_generator,
                steps_per_epoch=self.train_size // self.batch_size,
                epochs=self.n_iter + 1,
                initial_epoch=self.n_iter,
                shuffle=True,
                verbose=1
            )

            probas = self.model.predict_generator(
                valid_generator,
                steps=np.ceil(self.valid_size / self.batch_size)
            )
            valid_score = roc_auc_score(self.valid_y, probas, average='macro')

            self.n_iter += 1

            logger.info(
                f'valid_auc={valid_score:.3f}, '
                f'max_valid_auc={self.max_score:.3f}'
            )

            if self.max_score < valid_score:
                self.max_score = valid_score

                break

    def test(self, test_x, remaining_time_budget=None):
        if not hasattr(self, 'test_x'):
            self.test_x = get_kapre_logmel(
                test_x,
                len_sample=5,
                model=self.logmel_model
            )
            self.test_x = (
                self.test_x - np.mean(
                    self.test_x,
                    axis=(1, 2, 3),
                    keepdims=True
                )
            ) / np.std(self.test_x, axis=(1, 2, 3), keepdims=True)
            self.test_size, _, _, _ = self.test_x.shape

        probas = np.zeros(
            (self.metadata['test_num'], self.metadata['class_num'])
        )

        for _ in range(self.n_predictions):
            test_generator = TTAGenerator(
                self.test_x,
                batch_size=self.batch_size
            )()

            probas += self.model.predict_generator(
                test_generator,
                steps=np.ceil(self.test_size / self.batch_size)
            )

        probas /= self.n_predictions

        return probas
