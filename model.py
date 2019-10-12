import librosa
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
    config = tf.compat.v1.ConfigProto

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
            train_y = np.argmax(train_y, axis=1)

            classes = np.unique(train_y)
            class_weight = compute_class_weight('balanced', classes, train_y)

            self.class_weight = dict(zip(classes, class_weight))

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

        if self.n_iter < 9:
            train_size = 0.1 * (self.n_iter + 1)
            X, _, y, _ = train_test_split(
                X,
                y,
                random_state=self.random_state,
                shuffle=True,
                train_size=train_size
            )

        if not hasattr(self, 'model'):
            num_class = self.metadata['class_num']

            self.model = cnn_model(X.shape[1:], num_class)

            optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-06)

            self.model.compile(
                loss='sparse_categorical_crossentropy',
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

        self.model.fit_generator(
            datagen.flow(X, y, batch_size=32),
            callbacks=callbacks,
            class_weight=self.class_weight,
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
