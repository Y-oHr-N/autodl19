import os
import pickle

os.system("pip3 install -q lightgbm==2.3.1")
os.system("pip3 install -q scikit-learn==0.22")

import lightgbm as lgb
import pandas as pd

from models import LGBMRegressor
from preprocessing import Astype
from preprocessing import CalendarFeatures
from preprocessing import ClippedFeatures
from preprocessing import ModifiedSelectFromModel
from preprocessing import TargetShiftFeatures
from preprocessing import get_time_shift_range
from preprocessing import get_pred_time_diff
from preprocessing import Profiler


class Model:
    def __init__(self, info, test_timestamp, pred_timestamp):
        self.info = info
        self.label = info["label"]
        self.primary_id = info["primary_id"]
        self.categorical_cols = [
            col for col, types in info["schema"].items() if types == "str"
        ]
        self.numerical_cols = [
            col
            for col, types in info["schema"].items()
            if types == "num" and col != self.label
        ]
        self.time_cols = [info["primary_timestamp"]]
        self.update_interval = int(len(pred_timestamp) / 5)
        self.n_predict = 0
        self.shift_range = get_time_shift_range(
            pred_timestamp, info["primary_timestamp"]
        )
        self.pred_time_diff = get_pred_time_diff(
            pred_timestamp, info["primary_timestamp"]
        )

    def train(self, train_data, time_info):
        self.astype_ = Astype(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            time_cols=self.time_cols,
        )
        self.clipped_features_ = ClippedFeatures()
        self.target_shift_features = TargetShiftFeatures(
            shift_range=self.shift_range,
            pred_time_diff=self.pred_time_diff,
            primary_id=self.primary_id,
            time_col=self.time_cols[0],
        )
        self.calendar_features_ = CalendarFeatures(dtype="float32", encode=True)
        self.selector_ = ModifiedSelectFromModel(
            lgb.LGBMRegressor(importance_type="gain", random_state=0), threshold=1e-06
        )
        self.model_ = LGBMRegressor()

        X = train_data.sort_values(self.time_cols)
        y = X.pop(self.label)

        X = self.astype_.fit_transform(X)

        if len(self.numerical_cols) > 0:
            X[self.numerical_cols] = self.clipped_features_.fit_transform(
                X[self.numerical_cols]
            )

        self.target_shift_features.fit(X, y)

        X = self.target_shift_features.transform(X, istrain=True)

        Xt = self.calendar_features_.fit_transform(X[self.time_cols])
        X = X.drop(columns=self.time_cols)
        X = pd.concat([X, Xt], axis=1)
        X = self.selector_.fit_transform(X, y)

        self.model_.fit(X, y)

        return "predict"

    def predict(self, new_history, pred_record, time_info):
        X = self.astype_.transform(pred_record)

        if len(self.numerical_cols) > 0:
            X[self.numerical_cols] = self.clipped_features_.transform(
                X[self.numerical_cols]
            )

        X_new = new_history
        y_new = X_new.pop(self.label)

        self.target_shift_features.update(X_new, y_new)

        del X_new, y_new

        X = self.target_shift_features.transform(X, istrain=False)

        Xt = self.calendar_features_.transform(X[self.time_cols])
        X = X.drop(columns=self.time_cols)
        X = pd.concat([X, Xt], axis=1)
        X = self.selector_.transform(X)

        y_pred = self.model_.predict(X)

        self.n_predict += 1

        if self.n_predict > self.update_interval:
            next_step = "update"

            self.n_predict = 0

        else:
            next_step = "predict"

        return list(y_pred), next_step

    def update(self, train_data, test_history_data, time_info):
        total_data = pd.concat([train_data, test_history_data])

        self.train(total_data, time_info)

        next_step = "predict"

        return next_step

    def save(self, model_dir, time_info):
        pkl_list = []

        for attr in dir(self):
            if attr.startswith("__") or attr in [
                "train",
                "predict",
                "update",
                "save",
                "load",
            ]:
                continue

            pkl_list.append(attr)

            pickle.dump(
                getattr(self, attr), open(os.path.join(model_dir, f"{attr}.pkl"), "wb")
            )

        pickle.dump(pkl_list, open(os.path.join(model_dir, f"pkl_list.pkl"), "wb"))

    def load(self, model_dir, time_info):
        pkl_list = pickle.load(open(os.path.join(model_dir, "pkl_list.pkl"), "rb"))

        for attr in pkl_list:
            setattr(
                self,
                attr,
                pickle.load(open(os.path.join(model_dir, f"{attr}.pkl"), "rb")),
            )
