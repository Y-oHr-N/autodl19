import librosa
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import sequence

from keras.backend.tensorflow_backend import set_session

from sklearn.model_selection import train_test_split

try:
    config = tf.ConfigProto()
except AttributeError:
    config = tf.compat.v1.ConfigProto

config.gpu_options.allow_growth = True
config.log_device_placement = False
sess = tf.Session(config=config)

set_session(sess)


def extract_mfcc(data, sr=16_000):
    results = []

    for d in data:
        r = librosa.feature.mfcc(d, sr=sr, n_mfcc=24)
        r = r.transpose()

        results.append(r)

    return results


def pad_seq(data, pad_len):
    return sequence.pad_sequences(
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


class Model(object):
    def __init__(self, metadata):
        self.done_training = False
        self.metadata = metadata

        self.model = None
        self.max_len = None

        self.train_x = None
        self.train_y = None
        self.test_x = None

        self.iter = 1
        self.START = True
        self.random_state=0

    def train(self, train_dataset, remaining_time_budget=None):
        if remaining_time_budget <= self.metadata['time_budget']*0.125:
            self.done_training = True
            return
        if self.START:
            train_x, train_y = train_dataset
            fea_x = extract_mfcc(train_x)
            self.max_len = max([len(_) for _ in fea_x])
            fea_x = pad_seq(fea_x, self.max_len)
            self.train_x = fea_x[:, :, :, np.newaxis]
            self.train_y = train_y

        num_class = self.metadata['class_num']
        if self.iter < 10:
            X, _, y, _ = train_test_split(
                self.train_x,
                self.train_y,
                random_state=self.random_state,
                train_size=0.1*self.iter,
                shuffle=True
            )
        else:
            X = self.train_x
            y = self.train_y

        self.model = cnn_model(X.shape[1:], num_class)

        optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-06)

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        #self.model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10
            )
        ]

        self.model.fit(
            X,
            np.argmax(y, axis=1),
            epochs=self.iter,
            initial_epoch=self.iter-1,
            callbacks=callbacks,
            validation_split=0.1,
            verbose=1,
            batch_size=32,
            shuffle=True
        )


    def test(self, test_x, remaining_time_budget=None):
        if self.START:
            fea_x = extract_mfcc(test_x)
            fea_x = pad_seq(fea_x, self.max_len)
            self.test_x = fea_x[:, :, :, np.newaxis]
        self.iter += 1
        self.START = False
        return self.model.predict_proba(self.test_x)
