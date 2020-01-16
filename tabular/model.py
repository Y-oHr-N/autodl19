# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""

from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
tf.logging.set_verbosity(tf.logging.ERROR)


class Model(object):
    """Fully connected neural network with no hidden layer."""

    def __init__(self, metadata):
        """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
        self.done_training = False

        self.metadata = metadata
        self.output_dim = self.metadata.get_output_size()

        # Set batch size (for both training and testing)
        self.batch_size = 128
        # Get model function from class method below
        """
        model_fn = self.model_fn
        """
        # Change to True if you want to show device info at each operation
        log_device_placement = False
        session_config = tf.ConfigProto(log_device_placement=log_device_placement)
        # Classifier using model_fn (see below)
        #TODO
        """
        self.classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            config=tf.estimator.RunConfig(session_config=session_config),
        )
        """

        # Attributes for preprocessing
        self.default_image_size = (112, 112)
        self.default_num_frames = 10
        self.default_shuffle_buffer = 100

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.train_begin_times = []
        self.test_begin_times = []
        self.li_steps_to_train = []
        self.li_cycle_length = []
        self.li_estimated_time = []
        self.time_estimator = LinearRegression()
        # Critical number for early stopping
        # Depends on number of classes (output_dim)
        # see the function self.choose_to_stop_early() below for more details
        self.num_epochs_we_want_to_train = 70

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

    This method will be called REPEATEDLY during the whole training/predicting
    process. So your `train` method should be able to handle repeated calls and
    hopefully improve your model performance after each call.

    ****************************************************************************
    ****************************************************************************
    IMPORTANT: the loop of calling `train` and `test` will only run if
        self.done_training = False
      (the corresponding code can be found in ingestion.py, search
      'M.done_training')
      Otherwise, the loop will go on until the time budget is used up. Please
      pay attention to set self.done_training = True when you think the model is
      converged or when there is not enough time for next round of training.
    ****************************************************************************
    ****************************************************************************

    Args:
      dataset: a `tf.data.Dataset` object. Each of its examples is of the form
            (example, labels)
          where `example` is a dense 4-D Tensor of shape
            (sequence_size, row_count, col_count, num_channels)
          and `labels` is a 1-D Tensor of shape
            (output_dim,).
          Here `output_dim` represents number of classes of this
          multilabel classification task.

          IMPORTANT: some of the dimensions of `example` might be `None`,
          which means the shape on this dimension might be variable. In this
          case, some preprocessing technique should be applied in order to
          feed the training of a neural network. For example, if an image
          dataset has `example` of shape
            (1, None, None, 3)
          then the images in this datasets may have different sizes. On could
          apply resizing, cropping or padding in order to have a fixed size
          input tensor.

      remaining_time_budget: time remaining to execute train(). The method
          should keep track of its execution time to avoid exceeding its time
          budget. If remaining_time_budget is None, no time budget is imposed.
    """
        if self.done_training:
            return

        # Count examples on training set
        if not hasattr(self, "num_examples_train"):
            logger.info("Counting number of examples on train set.")
            iterator = dataset.make_one_shot_iterator()
            example, labels = iterator.get_next()
            sample_count = 0
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        sess.run(labels)
                        sample_count += 1
                    except tf.errors.OutOfRangeError:
                        break
            self.num_examples_train = sample_count
            logger.info(
                "Finished counting. There are {} examples for training set.".format(
                    sample_count
                )
            )
        # load X, y from dataset
        X, y = self.to_numpy(dataset, True)
        X = X.reshape(-1, X.shape[2])
        dataset = TabularDataset(X, y)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)
        # define model
        if not hasattr(self, "model"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = NeuralNetClassifier(
                feature_num=X.shape[1],
                output_dim=self.outputdim
            ).to(self.device)

        self.train_begin_times.append(time.time())
        #TODO
        if len(self.train_begin_times) >= 2:
            cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
            self.li_cycle_length.append(cycle_length)

        # Get number of steps to train according to some strategy
        steps_to_train = self.get_steps_to_train(remaining_time_budget)
        #TODO
        if steps_to_train <= 0:
            logger.info(
                "Not enough time remaining for training + test. "
                + "Skipping training..."
            )
            self.done_training = True
        elif self.choose_to_stop_early():
            logger.info(
                "The model chooses to stop further training because "
                + "The preset maximum number of epochs for training is "
                + "obtained: self.num_epochs_we_want_to_train = "
                + str(self.num_epochs_we_want_to_train)
            )
            self.done_training = True
        else:
            msg_est = ""
            if len(self.li_estimated_time) > 0:
                estimated_duration = self.li_estimated_time[-1]
                estimated_end_time = time.ctime(int(time.time() + estimated_duration))
                msg_est = (
                    "estimated time for training + test: "
                    + "{:.2f} sec, ".format(estimated_duration)
                )
                msg_est += "and should finish around {}.".format(estimated_end_time)
            logger.info(
                "Begin training for another {} steps...{}".format(
                    steps_to_train, msg_est
                )
            )

            # Prepare input function for training
            # train_input_fn = lambda: self.input_function(dataset, is_training=True)

            # Start training
            train_start = time.time()
            #TODO
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            self.model.train()
            for epoch in range(1):
                for train_X, train_y in dataloader:
                    train_X = train_X.to(self.device)
                    train_y = train_y.to(self.device)
                    preds = self.model(train_X)
                    loss = citerion(preds, train_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # self.classifier.train(input_fn=train_input_fn, steps=steps_to_train)
            train_end = time.time()

            # Update for time budget managing
            train_duration = train_end - train_start
            self.li_steps_to_train.append(steps_to_train)
            logger.info(
                "{} steps trained. {:.2f} sec used. ".format(
                    steps_to_train, train_duration
                )
                + "Now total steps trained: {}. ".format(sum(self.li_steps_to_train))
                + "Total time used for training + test: {:.2f} sec. ".format(
                    sum(self.li_cycle_length)
                )
            )

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """
        # Count examples on test set
        if not hasattr(self, "num_examples_test"):
            logger.info("Counting number of examples on test set.")
            iterator = dataset.make_one_shot_iterator()
            example, labels = iterator.get_next()
            sample_count = 0
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        sess.run(labels)
                        sample_count += 1
                    except tf.errors.OutOfRangeError:
                        break
            self.num_examples_test = sample_count
            logger.info(
                "Finished counting. There are {} examples for test set.".format(
                    sample_count
                )
            )
        X, _ = self.to_numpy(dataset, False)
        X = X.reshape(-1, X.shape[2])
        dataset = TabularDataset(X, None)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=False)
        test_begin = time.time()
        self.test_begin_times.append(test_begin)
        logger.info("Begin testing...")

        # Prepare input function for testing
        # test_input_fn = lambda: self.input_function(dataset, is_training=False)

        # Start testing (i.e. making prediction on test set)
        # test_results = self.classifier.predict(input_fn=test_input_fn)
        self.model.eval()
        predictions = np.empty((0, self.output_dim))
        for test_X in dataloader:
            test_X = test_X.to(self.device)
            output = F.Softmax(self.model(test_X)).numpy()
            predictions = pd.concat([predictions, output], axis=0)
            # predictions = torch.cat((predictions, preds.softmax()))
        predictions = [x["probabilities"] for x in test_results]
        predictions = np.array(predictions)
        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Duration used for test: {:2f}".format(test_duration)
        )
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

    # Model functions that contain info on neural network architectures
    # Several model functions are to be implemented, for different domains
    def model_fn(self, features, labels, mode):
        """Linear model (with no hidden layer).

    For more information on how to write a model function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
    """
        is_training = False
        keep_prob = 1
        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True
            keep_prob = 0.8

        input_layer = features

        # Replace missing values by 0
        mask = tf.is_nan(input_layer)
        input_layer = tf.where(mask, tf.zeros_like(input_layer), input_layer)

        # Sum over time axis
        input_layer = tf.reduce_sum(input_layer, axis=1)
        mask = tf.reduce_sum(1 - tf.cast(mask, tf.float32), axis=1)

        # Flatten
        input_layer = tf.layers.flatten(input_layer)
        mask = tf.layers.flatten(mask)
        f = input_layer.get_shape().as_list()[1]  # tf.shape(input_layer)[1]

        # Build network
        x = tf.layers.batch_normalization(input_layer, training=is_training)
        x = tf.nn.dropout(x, keep_prob)
        x_skip = self.fc(x, 256, is_training)
        x = self.fc(x_skip, 256, is_training)
        x = tf.nn.dropout(x, keep_prob)
        x = self.fc(x, 256, is_training) + x_skip
        x_mid = self.fc(x, f, is_training)

        x = self.fc(x_mid, 256, is_training)
        for i in range(3):
            x = self.fc(x, 256, is_training)

        logits = tf.layers.dense(inputs=x, units=self.output_dim)
        sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": sigmoid_tensor > 0.5,  # tf.argmax(input=logits, axis=1),
            # "classes": binary_predictions,
            # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": sigmoid_tensor,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        # For multi-label classification, a correct loss is sigmoid cross entropy
        # s = tf.shape(labels)
        # w = tf.ones(s) * tf.reduce_sum(labels) / tf.cast(tf.reduce_prod(s), tf.float32)
        # w = tf.where(labels>0, 1-w, w)
        loss_labels = tf.reduce_sum(
            sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        loss_reconst = tf.reduce_sum(mask * tf.abs(tf.subtract(input_layer, x_mid)))
        loss = loss_labels + loss_reconst

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step()
                )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        assert mode == tf.estimator.ModeKeys.EVAL
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"]
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
        )

    def fc(self, x, out_dim, is_training):
        x = tf.layers.dense(inputs=x, units=out_dim)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        return x

    def input_function(self, dataset, is_training):
        """Given `dataset` received by the method `self.train` or `self.test`,
    prepare input to feed to model function.

    For more information on how to write an input function, see:
      https://www.tensorflow.org/guide/custom_estimators#write_an_input_function
    """
        dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))

        if is_training:
            # Shuffle input examples
            dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
            # Convert to RepeatDataset to train for several epochs
            dataset = dataset.repeat()

        # Set batch size
        dataset = dataset.batch(batch_size=self.batch_size)

        iterator_name = "iterator_train" if is_training else "iterator_test"

        if not hasattr(self, iterator_name):
            self.iterator = dataset.make_one_shot_iterator()

        # iterator = dataset.make_one_shot_iterator()
        iterator = self.iterator
        example, labels = iterator.get_next()
        return example, labels

    def preprocess_tensor_4d(self, tensor_4d):
        """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
    non-fixed). The output tensor wil have fixed, known shape.

    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels]
          where some dimensions might be `None`.
    Returns:
      A 4-D Tensor with fixed, known shape.
    """
        tensor_4d_shape = tensor_4d.shape
        logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

        if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
            num_frames = tensor_4d_shape[0]
        else:
            num_frames = self.default_num_frames
        if tensor_4d_shape[1] > 0:
            new_row_count = tensor_4d_shape[1]
        else:
            new_row_count = self.default_image_size[0]
        if tensor_4d_shape[2] > 0:
            new_col_count = tensor_4d_shape[2]
        else:
            new_col_count = self.default_image_size[1]

        if not tensor_4d_shape[0] > 0:
            logger.info(
                "Detected that examples have variable sequence_size, will "
                + "randomly crop a sequence with num_frames = "
                + "{}".format(num_frames)
            )
            tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
        if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
            logger.info(
                "Detected that examples have variable space size, will "
                + "resize space axes to (new_row_count, new_col_count) = "
                + "{}".format((new_row_count, new_col_count))
            )
            tensor_4d = resize_space_axes(
                tensor_4d, new_row_count=new_row_count, new_col_count=new_col_count
            )
        logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
        return tensor_4d

    def get_steps_to_train(self, remaining_time_budget):
        """Get number of steps for training according to `remaining_time_budget`.

    The strategy is:
      1. If no training is done before, train for 10 steps (ten batches);
      2. Otherwise, double the number of steps to train. Estimate the time
         needed for training and test for this number of steps;
      3. Compare to remaining time budget. If not enough, stop. Otherwise,
         proceed to training/test and go to step 2.
    """
        if (
            remaining_time_budget is None
        ):  # This is never true in the competition anyway
            remaining_time_budget = 1200  # if no time limit is given, set to 20min

        # for more conservative estimation
        remaining_time_budget = min(
            remaining_time_budget - 60, remaining_time_budget * 0.95
        )

        if len(self.li_steps_to_train) == 0:
            return 10
        else:
            steps_to_train = self.li_steps_to_train[-1] * 2

            # Estimate required time using linear regression
            X = np.array(self.li_steps_to_train).reshape(-1, 1)
            Y = np.array(self.li_cycle_length)
            self.time_estimator.fit(X, Y)
            X_test = np.array([steps_to_train]).reshape(-1, 1)
            Y_pred = self.time_estimator.predict(X_test)

            estimated_time = Y_pred[0]
            self.li_estimated_time.append(estimated_time)

            if estimated_time >= remaining_time_budget:
                return 0
            else:
                return steps_to_train

    def age(self):
        return time.time() - self.birthday

    def choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
    process).
    """
        batch_size = self.batch_size
        num_examples = self.num_examples_train
        num_epochs = sum(self.li_steps_to_train) * batch_size / num_examples
        logger.info("Model already trained for {:.4f} epochs.".format(num_epochs))
        return (
            num_epochs > self.num_epochs_we_want_to_train
        )  # Train for at least certain number of epochs then stop

    def to_numpy(self, dataset, is_training):
        """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.
    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.
    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).
    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
        if is_training:
            subset = "train"
        else:
            subset = "test"
        attr_X = "X_{}".format(subset)
        attr_Y = "Y_{}".format(subset)

        # Only iterate the TF dataset when it's not done yet
        if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            X = []
            Y = []
            with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
                while True:
                    try:
                        example, labels = sess.run(next_element)
                        X.append(example)
                        Y.append(labels)
                    except tf.errors.OutOfRangeError:
                        break
            #TODO fill na
            np.nan_to_num(X, copy=False, nan=np.nanmean(X))
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            dtype = torch.float
            X = X.to(dtype)
            y = y.to(dtype)
            setattr(self, attr_X, X)
            setattr(self, attr_Y, Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        return X, Y

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.n = X.shape[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.y  is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class FullyConnectedModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim
    ):
        super(Model, self).__init__()
        self.linear_layer = nn.Linear(input_dim, output_dim)
        self.bn_layer = nn.BatchNorm1d(output_dim)

    def forward(self, X):
        X = self.linear_layer(X)
        X = F.relu(self.bn_layer(X))
        return X


class SkipNN(nn.Module):
    def __init__(
        self,
        input_dim
    ):
        super(Model, self).__init__()
        # linear layers
        self.fc_layer1 = FullyConnectedModule(input_dim, 256)
        self.fc_layer2 = FullyConnectedModule(256, 256)
        self.fc_layer3 = FullyConnectedModule(256, 256)
        self.fc_layer4 = FullyConnectedModule(256, input_dim)
        # Batch norm layers

        # dropout layers
        self.dropout_layer = nn.Dropout(0.5)
    def forward(self, X):
        X_skip = self.fc_layer1(X)
        X = self.fc_layer2(X)
        X = self.dropout_layer(X)
        X = self.fc_layer3(X) + X_skip
        X = self.fc_layer4(X)
        return X


class TabularNN(nn.Module):
    def __init__(
        self,
        feature_num=None,
        output_dim=None,
        layer_size=3,
        dropout_p=0.5
    ):
        super(Model, self).__init__()
        # Batch Norm layers
        self.first_bn_layer = nn.BatchNorm1d(feature_num)
        # drop out layers
        self.dropout_layer = nn.Dropout(dropout_p)
        # skip layer
        self.skip_layer = SkipNN(feature_num)
        # fully connected layers
        self.mid_FC_layer = FullyConnectedModule(feature_num, 256)
        self.FC_layers = nn.ModuleList(
            [
                FullyConnectedModule(256, 256) for i in range(layer_size)
            ]
        )
        self.last_lin_layers = nn.Linear(256, output_dim)

    def forward(self, X):
        X = self.first_bn_layer(X)
        X = self.dropout_layer(X)
        X = self.skip_layer(X)
        X = self.mid_FC_layer(X)
        for FC_layer in self.FC_layers:
            X = FC_layer(X)
        X = self.last_lin_layers(X)
        return X


def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
    """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
    labels = tf.cast(labels, dtype=tf.float32)
    relu_logits = tf.nn.relu(logits)
    exp_logits = tf.exp(-tf.abs(logits))
    sigmoid_logits = tf.log(1 + exp_logits)
    element_wise_xent = relu_logits - labels * logits + sigmoid_logits
    return element_wise_xent


def get_num_entries(tensor):
    """Return number of entries for a TensorFlow tensor.

  Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
  Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
  """
    tensor_shape = tensor.shape
    assert len(tensor_shape) > 1
    num_entries = 1
    for i in tensor_shape[1:]:
        num_entries *= int(i)
    return num_entries


def crop_time_axis(tensor_4d, num_frames, begin_index=None):
    """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
    # pad sequence if not long enough
    pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[0], 0)
    padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

    # If not given, randomly choose the beginning index of frames
    if not begin_index:
        maxval = tf.shape(padded_tensor)[0] - num_frames + 1
        begin_index = tf.random.uniform([1], minval=0, maxval=maxval, dtype=tf.int32)
        begin_index = tf.stack([begin_index[0], 0, 0, 0], name="begin_index")

    sliced_tensor = tf.slice(
        padded_tensor, begin=begin_index, size=[num_frames, -1, -1, -1]
    )

    return sliced_tensor


def resize_space_axes(tensor_4d, new_row_count, new_col_count):
    """Given a 4-D tensor, resize space axes to have target size.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
    resized_images = tf.image.resize_images(
        tensor_4d, size=(new_row_count, new_col_count)
    )
    return resized_images


def get_logger(verbosity_level):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")
