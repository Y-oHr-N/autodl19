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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
import lightgbm as lgb

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
        self.feature_size = self.metadata.get_matrix_size()[1]

        # Set batch size (for both training and testing)
        self.batch_size = 128
        # Change to True if you want to show device info at each operation
        log_device_placement = False
        session_config = tf.ConfigProto(log_device_placement=log_device_placement)
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
        self.epoch_num = 0
        self.max_score = 0
        self.num_epochs_we_want_to_train = 70
        self.is_first = True
        self.early_stopping_rounds = 10
        self.lgb_weight = 0.8
        self.using_model = "NN"

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
        # load X, y from dataset
        if self.is_first:
            X, y = self.to_numpy(dataset, True)
            self.num_examples_train = X.shape[0]
            print("train size: ", X.shape)
            X = np.nan_to_num(X)
            if not hasattr(self, "is_multi_label"):
                if np.sum(y) != self.num_examples_train:
                    self.is_multi_label = True
                else:
                    self.is_multi_label = False
            if not hasattr(self, "cat_cols"):
                self.cat_cols = self.get_cat_cols(X)
                logger.info(
                "category estimate"
                )
                print(self.cat_cols)
                self.emb_dims = []
                self.label_encoders = []
                for i in self.cat_cols:
                    emb_dim = len(np.unique(X[:, i]))
                    # self.emb_dims.append(((emb_dim, int(6*(emb_dim**(1/4)))  )))
                    self.emb_dims.append((emb_dim, int(max(2, min(emb_dim / 2, 50)))))
                    label_encoder = LabelEncoder()
                    label_encoder.fit(X[:, i])
                    self.label_encoders.append(label_encoder)
                self.standard_scaler = StandardScaler()
                self.standard_scaler.fit(X)
                self.no_of_numerical = X.shape[1]
                self.lin_layer_sizes = [256, 256]
                self.emb_dropout = 0.5
                self.lin_layer_dropouts = [0.5, 0.5]

            #TODO fill na
            #TODO feature engineering
            #TODO estimate type (categorical or numerical)

            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
                X,
                y,
                random_state=42,
                shuffle=True,
                stratify=np.argmax(y, axis=1),
                train_size=0.9
            )
            # define dataset and dataloader
            # dataset_train = TabularDataset(X_train, y_train)
            # dataset_valid = TabularDataset(X_valid, y_valid)
            self.dataset_train = TabularEmbeddingDataset(
                self.X_train,
                self.y_train,
                cat_cols=self.cat_cols,
                standard_scaler=self.standard_scaler,
                label_encoder=self.label_encoders
            )
            self.dataset_valid = TabularEmbeddingDataset(
                self.X_valid,
                self.y_valid,
                cat_cols=self.cat_cols,
                standard_scaler=self.standard_scaler,
                label_encoder=self.label_encoders
            )

            self.dataloader_train = DataLoader(self.dataset_train, self.batch_size, shuffle=True)
            self.dataloader_valid = DataLoader(self.dataset_valid, self.batch_size, shuffle=False)
            if self.is_multi_label:
                self.rf_models = []
                self.predictions_rf_valid = np.empty((self.X_valid.shape[0], 0))
                for i in range(self.output_dim):
                    rf_model = lgb.LGBMClassifier(
                        boosting_type="rf",
                        objective="binary",
                        num_leaves=2**5-1,
                        max_depth=5,
                        # learning_rate=0.1,
                        n_estimators=100,
                        colsample_bytree=0.5,
                        subsample=0.5,
                        subsample_freq=1,
                    )
                    rf_model.fit(
                        self.X_train,
                        self.y_train[:,i],
                        eval_set=[(self.X_valid, self.y_valid[:,i])],
                        eval_metric="logloss",
                        early_stopping_rounds=10,
                        verbose=100,
                    )

                    pred_tmp = rf_model.predict_proba(self.X_valid)[:,1].reshape(-1, 1)
                    self.predictions_rf_valid = np.concatenate([self.predictions_rf_valid, pred_tmp], axis=1)
                    self.rf_models.append(rf_model)
            else:
                if self.output_dim == 2:
                    self.rf_model = lgb.LGBMClassifier(
                        boosting_type="rf",
                        objective="binary",
                        num_leaves=2**5-1,
                        max_depth=5,
                        # learning_rate=0.1,
                        n_estimators=100,
                        colsample_bytree=0.5,
                        subsample=0.5,
                        subsample_freq=1,
                        )
                    self.rf_model.fit(
                        self.X_train,
                        np.argmax(self.y_train, axis=1),
                        eval_set=[(self.X_valid, np.argmax(self.y_valid, axis=1))],
                        eval_metric="logloss",
                        early_stopping_rounds=10,
                        verbose=100
                        )
                else:
                    self.rf_model = lgb.LGBMClassifier(
                        boosting_type="rf",
                        objective="multiclass",
                        num_leaves=2**5-1,
                        max_depth=5,
                        # learning_rate=0.1,
                        n_estimators=100,
                        colsample_bytree=0.5,
                        subsample=0.5,
                        subsample_freq=1,
                        num_class=self.output_dim
                        )
                    self.rf_model.fit(
                        self.X_train,
                        np.argmax(self.y_train, axis=1),
                        eval_set=[(self.X_valid, np.argmax(self.y_valid, axis=1))],
                        eval_metric="multi_logloss",
                        early_stopping_rounds=10,
                        verbose=100
                        )
                self.predictions_rf_valid = self.rf_model.predict_proba(self.X_valid)
            self.valid_score_rf = 2 * roc_auc_score(self.y_valid, self.predictions_rf_valid, average="macro") - 1
            print("rf_auc: ", self.valid_score_rf)
            return self

        # define model
        if not hasattr(self, "model"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            """
            self.model = TabularNN(
                feature_num=X.shape[1],
                output_dim=self.output_dim
            ).to(self.device)
            """
            self.model = TabularEmbeddingNN(
                emb_dims=self.emb_dims,
                no_of_numerical=self.no_of_numerical,
                lin_layer_sizes=self.lin_layer_sizes,
                output_size=self.output_dim,
                emb_dropout=self.emb_dropout,
                lin_layer_dropouts=self.lin_layer_dropouts
            ).to(self.device)
            if self.is_multi_label:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.train_begin_times.append(time.time())
        #TODO time management
        if len(self.train_begin_times) >= 2:
            cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
            self.li_cycle_length.append(cycle_length)

        # Get number of steps to train according to some strategy
        steps_to_train = self.get_steps_to_train(remaining_time_budget)
        #TODO time management and early stopping
        if steps_to_train <= 0:
            logger.info(
                "Not enough time remaining for training + test. "
                + "Skipping training..."
            )
            self.done_training = True
        # elif self.choose_to_stop_early():
        #     logger.info(
        #         "The model chooses to stop further training because "
        #         + "The preset maximum number of epochs for training is "
        #         + "obtained: self.num_epochs_we_want_to_train = "
        #         + str(self.num_epochs_we_want_to_train)
        #    )
        #     self.done_training = True
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

            # Start training
            train_start = time.time()
            #TODO corresponding multi class
            for i in range(self.early_stopping_rounds):
                running_loss = 0.0
                auc = []
                self.model.train()
                for X_num_batch, X_cat_batch, y_batch in self.dataloader_train:
                    X_num_batch = X_num_batch.to(self.device)
                    X_cat_batch = X_cat_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = self.model(X_num_batch, X_cat_batch)
                    if self.is_multi_label:
                        loss = self.criterion(preds, y_batch)
                    else:
                        loss = self.criterion(preds, torch.argmax(y_batch, dim=1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item() / self.dataset_train.__len__()

                self.model.eval()
                self.predictions_valid = np.empty((0, self.output_dim))
                for X_num_batch, X_cat_batch, y_batch in self.dataloader_valid:
                    X_num_batch = X_num_batch.to(self.device)
                    X_cat_batch = X_cat_batch.to(self.device)
                    if self.is_multi_label:
                        output = torch.sigmoid(self.model(X_num_batch, X_cat_batch)).data.cpu().numpy()
                    else:
                        output = F.softmax(self.model(X_num_batch, X_cat_batch)).data.cpu().numpy()
                    self.predictions_valid = np.concatenate([self.predictions_valid, output], axis=0)
                valid_score = 2 * roc_auc_score(self.y_valid, self.predictions_valid, average="macro") - 1
                print("loss : ", running_loss)
                print("auc : ", valid_score)
                self.epoch_num += 1
                if self.max_score < valid_score:
                    self.max_score = valid_score
                    break
                if self.epoch_num <= 5:
                    break
                if i == self.early_stopping_rounds - 1:
                    self.done_training = True

            train_end = time.time()
        if self.done_training:
            if self.is_multi_label:
                self.lgb_models = []
                predictions_lgb = np.empty((self.X_valid.shape[0], 0))
                for i in range(self.output_dim):
                    lgb_model = lgb.LGBMClassifier(
                        boosting_type="gbdt",
                        objective="binary",
                        num_leaves=2**5-1,
                        max_depth=5,
                        learning_rate=0.1,
                        n_estimators=1000,
                        colsample_bytree=0.8,
                        subsample=0.8,
                        subsample_freq=1
                    )
                    lgb_model.fit(
                        self.X_train,
                        self.y_train[:,i],
                        eval_set=[(self.X_valid, self.y_valid[:,i])],
                        eval_metric="logloss",
                        early_stopping_rounds=10,
                        verbose=100
                    )

                    pred_tmp = lgb_model.predict_proba(self.X_valid)[:,1].reshape(-1, 1)
                    predictions_lgb = np.concatenate([predictions_lgb, pred_tmp], axis=1)
                    self.lgb_models.append(lgb_model)
            else:
                if self.output_dim == 2:
                    self.lgb_model = lgb.LGBMClassifier(
                        boosting_type="gbdt",
                        objective="binary",
                        num_leaves=2**5-1,
                        max_depth=5,
                        learning_rate=0.1,
                        n_estimators=1000,
                        colsample_bytree=0.8,
                        subsample=0.8,
                        subsample_freq=1,
                        )
                    self.lgb_model.fit(
                        self.X_train,
                        np.argmax(self.y_train, axis=1),
                        eval_set=[(self.X_valid, np.argmax(self.y_valid, axis=1))],
                        eval_metric="logloss",
                        early_stopping_rounds=10,
                        verbose=100
                        )
                else:
                    self.lgb_model = lgb.LGBMClassifier(
                        boosting_type="gbdt",
                        objective="multiclass",
                        num_leaves=2**5-1,
                        max_depth=5,
                        learning_rate=0.1,
                        n_estimators=1000,
                        colsample_bytree=0.8,
                        subsample=0.8,
                        subsample_freq=1,
                        num_class=self.output_dim
                        )
                    self.lgb_model.fit(
                        self.X_train,
                        np.argmax(self.y_train, axis=1),
                        eval_set=[(self.X_valid, np.argmax(self.y_valid, axis=1))],
                        eval_metric="multi_logloss",
                        early_stopping_rounds=10,
                        verbose=100
                        )
                predictions_lgb = self.lgb_model.predict_proba(self.X_valid)
            valid_score_lgb = 2 * roc_auc_score(self.y_valid, predictions_lgb, average="macro") - 1
            print("lgb_auc: ", valid_score_lgb)
            predictions_ensemble = (1-self.lgb_weight)*self.predictions_valid + self.lgb_weight*predictions_lgb
            valid_score_ensemble = 2 * roc_auc_score(self.y_valid, predictions_ensemble, average="macro") - 1
            print("ensemble_auc: ", valid_score_ensemble)
            if valid_score_lgb > valid_score_ensemble:
                self.using_model = "lgb"
            else:
                self.using_model = "ensemble"
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
        if self.is_first:
            self.X_test, _ = self.to_numpy(dataset, False)
            self.X_test = np.nan_to_num(self.X_test)
            self.dataset_test = TabularEmbeddingDataset(
                self.X_test,
                None,
                cat_cols=self.cat_cols,
                standard_scaler=self.standard_scaler,
                label_encoder=self.label_encoders
            )
            self.dataloader_test = DataLoader(self.dataset_test, self.batch_size, shuffle=False)
            if self.is_multi_label:
                self.predictions_rf_test = np.empty((self.X_test.shape[0], 0))
                for rf_model in self.rf_models:
                    pred_tmp = rf_model.predict_proba(self.X_test)[:,1].reshape(-1, 1)
                    self.predictions_rf_test = np.concatenate([self.predictions_rf_test, pred_tmp], axis=1)
            else:
                self.predictions_rf_test = self.rf_model.predict_proba(self.X_test)
            print(self.predictions_rf_test.shape)
            print(self.X_test.shape)
            self.is_first = False
            return self.predictions_rf_test
        test_begin = time.time()
        self.test_begin_times.append(test_begin)
        logger.info("Begin testing...")

        # Prepare input function for testing

        # Start testing (i.e. making prediction on test set)
        self.model.eval()
        if not self.done_training:
            self.predictions_test = np.empty((0, self.output_dim))
            for X_num_batch, X_cat_batch, in self.dataloader_test:
                X_num_batch = X_num_batch.to(self.device)
                X_cat_batch = X_cat_batch.to(self.device)
                if self.is_multi_label:
                    output = torch.sigmoid(self.model(X_num_batch, X_cat_batch)).data.cpu().numpy()
                else:
                    output = F.softmax(self.model(X_num_batch, X_cat_batch)).data.cpu().numpy()
                self.predictions_test = np.concatenate([self.predictions_test, output], axis=0)
        if self.using_model == "NN":
            if self.max_score > self.valid_score_rf:
                predictions_ensemble_rf = 0.8*self.predictions_valid + (1-0.8)*self.predictions_rf_valid
                if self.max_score < 2 * roc_auc_score(self.y_valid, predictions_ensemble_rf, average="macro") - 1:
                    self.predictions_test = 0.8*self.predictions_test + (1-0.8)*self.predictions_rf_test
                    print("ensemble auc: ", 2 * roc_auc_score(self.y_valid, predictions_ensemble_rf, average="macro") - 1)
            else:
                predictions_ensemble_rf = 0.2*self.predictions_valid + (1-0.2)*self.predictions_rf_valid
                if self.valid_score_rf < 2 * roc_auc_score(self.y_valid, predictions_ensemble_rf, average="macro") - 1:
                    self.predictions_test = 0.2*self.predictions_test + (1-0.2)*self.predictions_rf_test
                    print("ensemble auc: ", 2 * roc_auc_score(self.y_valid, predictions_ensemble_rf, average="macro") - 1)
                else:
                    self.predictions_test = self.predictions_rf_test
        else:
            if self.is_multi_label:
                predictions_lgb = np.empty((self.X_test.shape[0], 0))
                for lgb_model in self.lgb_models:
                    pred_tmp = lgb_model.predict_proba(self.X_test)[:,1].reshape(-1, 1)
                    predictions_lgb = np.concatenate([predictions_lgb, pred_tmp], axis=1)
            else:
                predictions_lgb = self.lgb_model.predict_proba(self.X_test)
            if self.using_model == "lgb":
                self.predictions_test = predictions_lgb
            elif self.using_model == "ensemble":
                self.predictions_test = (1-self.lgb_weight)*self.predictions_test + self.lgb_weight*predictions_lgb

        test_end = time.time()

        # Update some variables for time management
        test_duration = test_end - test_begin
        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Duration used for test: {:2f}".format(test_duration)
        )
        return self.predictions_test

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
        if remaining_time_budget < 600:
            return 0
        # for more conservative estimation
        remaining_time_budget = min(
            remaining_time_budget - 60, remaining_time_budget * 0.6
        )

        if len(self.li_steps_to_train) == 0:
            return 1
        else:
            steps_to_train = self.li_steps_to_train[-1] + 1

            # Estimate required time using linear regression
            #X = np.array(self.li_steps_to_train).reshape(-1, 1)
            #Y = np.array(self.li_cycle_length)
            #self.time_estimator.fit(X, Y)
            #X_test = np.array([steps_to_train]).reshape(-1, 1)
            #Y_pred = self.time_estimator.predict(X_test)

            #estimated_time = Y_pred[0]
            estimated_time = self.li_cycle_length[0]
            self.li_estimated_time.append(estimated_time)
            if self.early_stopping_rounds*estimated_time >= remaining_time_budget:
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
            dataset = dataset.batch(batch_size=1024, drop_remainder=False)
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            X = np.empty((0, self.feature_size))
            Y = np.empty((0, self.output_dim))
            with tf.Session(config=tf.ConfigProto(log_device_placement=False,
                gpu_options=tf.GPUOptions(
                    allow_growth=True
                )
            )) as sess:
                while True:
                    try:
                        example, labels = sess.run(next_element)
                        example = example.reshape(-1, self.feature_size)
                        labels = labels.reshape(-1, self.output_dim)
                        X = np.append(X, example, axis=0)
                        Y = np.append(Y, labels, axis=0)
                    except tf.errors.OutOfRangeError:
                        break
            setattr(self, attr_X, X)
            setattr(self, attr_Y, Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        return X, Y

    def get_cat_cols(self, X):
        cat_cols = []
        for i in range(X.shape[1]):
            unique_list = np.unique(X[:, i])
            if ((np.min(unique_list) == 0) | (np.min(unique_list) == 1))\
                & (np.all(np.diff(unique_list, n=1) == 1)):
                cat_cols.append(i)
        return cat_cols

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.n = X.shape[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.y  is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

class TabularEmbeddingDataset(Dataset):
    def __init__(self, X, y, cat_cols=None, standard_scaler=None, label_encoder=None):
        self.n = X.shape[0]
        self.y = y
        self.cat_cols = cat_cols if cat_cols else []
        self.numerical_cols = list(range(X.shape[1]))

        if self.numerical_cols:
            self.numerical_X = X
            self.numerical_X = standard_scaler.transform(self.numerical_X)
        else:
            self.numerical_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = X[:, self.cat_cols]
            for i in range(self.cat_X.shape[1]):
                self.cat_X[:,i] = label_encoder[i].transform(self.cat_X[:,i])
        else:
            self.cat_X = np.zeros((self.n, 1))
        self.numerical_X = torch.Tensor(self.numerical_X)
        self.cat_X = torch.Tensor(self.cat_X)

        self.numerical_X = self.numerical_X.to(torch.float)
        self.cat_X = self.cat_X.to(torch.long)
        if y is not None:
            self.y = torch.Tensor(self.y)
            self.y = self.y.to(torch.float)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.y is not None:
            return [self.numerical_X[idx], self.cat_X[idx], self.y[idx]]
        else:
            return [self.numerical_X[idx], self.cat_X[idx]]

class FullyConnectedModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim
    ):
        super(FullyConnectedModule, self).__init__()
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
        super(SkipNN, self).__init__()
        # linear layers
        self.fc_layer1 = FullyConnectedModule(input_dim, 256)
        self.fc_layer2 = FullyConnectedModule(256, 256)
        self.fc_layer3 = FullyConnectedModule(256, 256)
        self.fc_layer4 = FullyConnectedModule(256, input_dim)
        # Batch norm layers

        # dropout layers
        self.dropout_layer = nn.Dropout(0.5)
    def forward(self, X):
        X_skip = self.fc_layer1(X) # input_dim -> 256
        X = self.fc_layer2(X_skip) # 256 -> 256
        X = self.dropout_layer(X) # 256
        X = self.fc_layer3(X) + X_skip # 256 -> 256
        X = self.fc_layer4(X) # 256 -> input_dim
        return X


class TabularNN(nn.Module):
    def __init__(
        self,
        feature_num=None,
        output_dim=None,
        layer_size=3,
        dropout_p=0.5
    ):
        super(TabularNN, self).__init__()
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
        X = self.first_bn_layer(X) # input_dim
        X = self.dropout_layer(X) # input_dim
        X = self.skip_layer(X) # input_dim -> input_dim
        X = self.mid_FC_layer(X) # input_dim -> 256
        for FC_layer in self.FC_layers:
            X = FC_layer(X) # 256 -> 256
        X = self.last_lin_layers(X) # 256 -> output_dim
        return X

class TabularEmbeddingNN(nn.Module):
    def __init__(
        self,
        emb_dims,
        no_of_numerical,
        lin_layer_sizes,
        output_size,
        emb_dropout,
        lin_layer_dropouts
    ):
        super(TabularEmbeddingNN, self).__init__()

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])

        self.no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_numerical = no_of_numerical
        # self.skip_layer = SkipNN(self.no_of_embs + self.no_of_numerical)
        first_lin_layer = nn.Linear(
            self.no_of_embs + self.no_of_numerical, lin_layer_sizes[0]
        )
        self.lin_layers = nn.ModuleList(
            [first_lin_layer] +
            [
                nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                for i in range(len(lin_layer_sizes) - 1)
            ]
        )
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.first_bn_layer = nn.BatchNorm1d(self.no_of_numerical)
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        )

        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(size) for size in lin_layer_dropouts]
        )

    def forward(self, numerical_data, cat_data):
        if self.no_of_embs != 0:
            X = [
                emb_layer(cat_data[:, i]) for i, emb_layer in enumerate(self.emb_layers)
            ]
            X = torch.cat(X, 1)
            X = self.emb_dropout_layer(X)
        if self.no_of_numerical != 0:
            normalized_numerical_data = self.first_bn_layer(numerical_data)

            if self.no_of_embs != 0:
                X = torch.cat([X, normalized_numerical_data], 1)
            else:
                X = normalized_numerical_data
        # X = self.skip_layer(X)
        for lin_layer, dropout_layer, bn_layer in zip(
            self.lin_layers, self.dropout_layers, self.bn_layers
        ):
            X = F.relu(lin_layer(X))
            X = bn_layer(X)
            X = dropout_layer(X)

        X = self.output_layer(X)

        return X

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
