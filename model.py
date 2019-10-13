import librosa
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
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

        self.sample_num = len(X_train)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (2 * self.batch_size))

            for i in range(itr_num):
                batch_ids = indexes[
                    2 * i * self.batch_size:2 * (i + 1) * self.batch_size
                ]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))

        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


class Model(object):
    def __init__(self, metadata, random_state=0):
        self.metadata = metadata
        self.random_state = random_state

        self.done_training = False
        self.n_iter = 0

    def train(self, train_dataset, remaining_time_budget=None):
        if remaining_time_budget <= 0.125 * self.metadata['time_budget']:
            self.done_training = True

            return

        if not hasattr(self, 'train_x'):
            train_x, train_y = train_dataset
            fea_x = extract_mfcc(train_x)

            self.max_len = max([len(_) for _ in fea_x])

            fea_x = pad_seq(fea_x, self.max_len)
            train_x = fea_x[:, :, :, np.newaxis]

            self.train_x, self.val_x, self.train_y, self.val_y = \
                train_test_split(
                    train_x,
                    train_y,
                    random_state=self.random_state,
                    shuffle=True,
                    train_size=0.9
                )

        X = self.train_x
        y = self.train_y

        if not hasattr(self, 'model'):
            num_class = self.metadata['class_num']

            self.model = cnn_model(X.shape[1:], num_class)

            optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-06)

            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

        # self.model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ]
        datagen = ImageDataGenerator(
            preprocessing_function=get_frequency_masking()
        )
        training_generator = MixupGenerator(
            X,
            y,
            alpha=0.2,
            batch_size=32,
            datagen=datagen
        )()

        self.model.fit_generator(
            training_generator,
            steps_per_epoch=X.shape[0] // 32,
            callbacks=callbacks,
            epochs=self.n_iter + 1,
            initial_epoch=self.n_iter,
            shuffle=True,
            validation_data=(self.val_x, self.val_y),
            verbose=1
        )

        self.n_iter += 1

    def test(self, test_x, remaining_time_budget=None):
        if not hasattr(self, 'test_x'):
            fea_x = extract_mfcc(test_x)
            fea_x = pad_seq(fea_x, self.max_len)

            self.test_x = fea_x[:, :, :, np.newaxis]

        return self.model.predict_proba(self.test_x)
