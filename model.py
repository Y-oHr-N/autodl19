import logging

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf

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
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

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

# parameters
SAMPLING_FREQ = 16_000
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1_024  # 0.064 sec
FMIN = 20
FMAX = SAMPLING_FREQ // 2


def logmelspectrogram(X):
    melspec = librosa.feature.melspectrogram(
        X,
        sr=SAMPLING_FREQ,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    ).astype(np.float32)

    return librosa.power_to_db(melspec)


def get_num_frame(n_sample, n_fft, n_shift):
    return np.floor(2.0 * (n_sample / n_fft) + 1.0).astype(np.int32)


def crop_time(
    X,  # array: len
    len_sample=5,  # 1 サンプル当たりの長さ [sec]
    min_sample=5,  # 切り出すサンプル数の最小個数
    max_sample=10,  # 切り出すサンプルの最大個数
):
    if len(X) < len_sample * max_sample * SAMPLING_FREQ:
        XX = X
    else:
        # TODO(Ishimura): 本当はシフト幅分余分にとりたい
        XX = X[:len_sample * max_sample * SAMPLING_FREQ]

    X = logmelspectrogram(XX)
    n_samples, n_features = X.shape

    # len_sample に対応する frame 長
    n_frame = get_num_frame(
        n_sample=len_sample * SAMPLING_FREQ,
        n_fft=N_FFT,
        n_shift=HOP_LENGTH
    )

    # データの frame 数（X.shape[1]）が len_sample * min_sample に満たない場合の repeat 数
    n_repeat = np.ceil(n_frame * min_sample / n_features).astype(np.int32)
    X_repeat = np.zeros([n_samples, n_features * n_repeat], np.float32)

    for i in range(n_repeat):
        X_repeat[:, i * n_features: (i + 1) * n_features] = X

    # 最低限, min_sample を確保
    if n_features <= n_frame * min_sample:
        n_sample = min_sample
    elif (n_frame * min_sample) < n_features <= (n_frame * max_sample):
        n_sample = (n_features // n_frame).astype(np.int32)
    else:
        n_sample = max_sample

    # Make New log-mel spectrogram
    X_new = np.zeros([n_samples, n_frame, n_sample], np.float32)

    for i in range(n_sample):
        X_new[:, :, i] = X_repeat[:, i * n_frame: (i + 1) * n_frame]

    return X_new


def make_cropped_dataset_5sec(
    X_list,  # list(array):
    y_list=None,  # array: n_sample x dim_label
    len_sample=5,  # 1 サンプル当たりの長さ [sec]
    min_sample=1,  # 切り出すサンプル数の最小個数
    max_sample=1,  # 切り出すサンプルの最大個数
):
    # さしあたり min_sample == max_sample == 1
    # -> y_results.shape == (len(X_list), dim_label) かつ，len(X_results) == len(X_list)
    X_results = []

    if y_list is not None:
        y_results = np.zeros([len(X_list), y_list.shape[1]], np.float32)

    for i in range(len(X_list)):
        logmels = crop_time(
            X_list[i],
            len_sample=len_sample,
            min_sample=min_sample,
            max_sample=max_sample
        )

        X_results.append(logmels[:, :, 0].T)

        if y_list is not None:
            y_results[i, :] = y_list[i, :]

    if y_list is None:
        y_results = None

    return X_results, y_results


def pad_seq(data, pad_len):
    return pad_sequences(
        data,
        dtype=np.float32,
        maxlen=pad_len,
        padding='post'
    )


def get_crop_image(image):
    time_dim, base_dim, _ = image.shape
    crop = np.random.randint(0, time_dim - base_dim)
    image = image[crop:crop+base_dim, :,:]

    return image


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

        f = np.random.randint(0, int(img_w * F))
        f0 = np.random.randint(0, img_w - f)
        c = input_img.mean()

        input_img[:, f0:f0 + f, :] = c

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
        self.max_auc = 0
        self.n_iter = 0
        self.not_improve_learning_iter = 0

    def train(self, train_dataset, remaining_time_budget=None):
        if remaining_time_budget <= 0.125 * self.metadata['time_budget']:
            self.done_training = True

            return

        if not hasattr(self, 'train_x'):
            n_classes = self.metadata['class_num']
            train_x, train_y = train_dataset
            fea_x, train_y = make_cropped_dataset_5sec(train_x, train_y)

            self.max_len = max([len(_) for _ in fea_x])

            fea_x = pad_seq(fea_x, self.max_len)
            train_x = fea_x[:, :, :, np.newaxis]

            logger.info(f'max_len={self.max_len}')
            logger.info(f'X.shape={train_x.shape}')

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
            self.model = cnn_model((w, w, 1), n_classes)

            optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-06)

            self.model.compile(optimizer, 'categorical_crossentropy')

        datagen = ImageDataGenerator(
            preprocessing_function=get_frequency_masking()
        )
        training_generator = MixupGenerator(
            self.train_x,
            self.train_y,
            batch_size=self.batch_size,
            datagen=datagen
        )()

        self.model.fit_generator(
            training_generator,
            steps_per_epoch=self.train_size // self.batch_size,
            epochs=self.n_iter + 1,
            initial_epoch=self.n_iter,
            shuffle=True,
            verbose=1
        )

        self.n_iter += 1

        valid_generator = TTAGenerator(
            self.valid_x,
            batch_size=self.batch_size
        )()
        valid_res = self.model.predict_generator(
            valid_generator,
            steps=np.ceil(self.valid_size / self.batch_size)
        )
        valid_auc = roc_auc_score(self.valid_y, valid_res, average='macro')

        logger.info(f'valid_auc={valid_auc:.3f}, max_auc={self.max_auc:.3f}')

        if self.max_auc < valid_auc:
            self.not_improve_learning_iter = 0
            self.max_auc = valid_auc
        else:
            self.not_improve_learning_iter += 1

        if self.not_improve_learning_iter >= self.patience:
            self.done_training = True

    def test(self, test_x, remaining_time_budget=None):
        if not hasattr(self, 'test_x'):
            fea_x, _ = make_cropped_dataset_5sec(test_x)
            fea_x = pad_seq(fea_x, self.max_len)

            self.test_x = fea_x[:, :, :, np.newaxis]
            self.test_size, _, _, _ = self.test_x.shape

        if self.not_improve_learning_iter == 0:
            test_generator = TTAGenerator(
                self.test_x,
                batch_size=self.batch_size
            )()

            self.probas = np.zeros(
                (self.metadata['test_num'], self.metadata['class_num'])
            )

            for _ in range(self.n_predictions):
                self.probas += self.model.predict_generator(
                    test_generator,
                    steps=np.ceil(self.test_size / self.batch_size)
                )

            self.probas /= self.n_predictions

        return self.probas
