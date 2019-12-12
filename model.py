import os
import pickle

os.system("pip3 install -q lightgbm==2.3.1")
os.system("pip3 install -q scikit-learn==0.22")

import lightgbm as lgb
import pandas as pd

from sklearn.feature_selection import SelectFromModel

from models import LGBMRegressor
from preprocessing import CalendarFeatures
from preprocessing import ClippedFeatures
from preprocessing import TypeAdapter
from preprocessing import Profiler


class Model:
    def __init__(self, info, test_timestamp, pred_timestamp):
        self.info = info
        self.primary_timestamp = info["primary_timestamp"]
        self.primary_id = info["primary_id"]
        self.label = info["label"]
        self.schema = info["schema"]

        print(f"\ninfo: {self.info}")

        self.dtype_cols = {}
        self.dtype_cols["cat"] = [
            col for col, types in self.schema.items() if types == "str"
        ]
        self.dtype_cols["num"] = [
            col for col, types in self.schema.items() if types == "num" and col != self.label
        ]

        self.test_timestamp = test_timestamp
        self.pred_timestamp = pred_timestamp

        self.n_test_timestamp = len(pred_timestamp)
        self.update_interval = int(self.n_test_timestamp / 5)

        print(f"sample of test record: {len(test_timestamp)}")
        print(f"number of pred timestamp: {len(pred_timestamp)}")

        self.lgb_model = LGBMRegressor()
        self.n_predict = 0

        print(f"Finish init\n")

    def train(self, train_data, time_info):
        print(f"\nTrain time budget: {time_info['train']}s")

        X = train_data
        y = train_data.pop(self.label)

        profiler = Profiler(self.primary_id)
        profiler.fit(X, y)

        # type adapter
        self.type_adapter = TypeAdapter(self.dtype_cols["cat"])
        X = self.type_adapter.fit_transform(X)

        # Clip numerical features
        if len(self.dtype_cols["num"]) > 0:
            self.clipped_features = ClippedFeatures()

            X.loc[:, self.dtype_cols["num"]] = self.clipped_features.fit_transform(
                X.loc[:, self.dtype_cols["num"]]
            )

        # parse time feature
        X[self.primary_timestamp] = pd.to_datetime(X[self.primary_timestamp], unit="s")
        self.calendar_features = CalendarFeatures()
        time_fea = self.calendar_features.fit_transform(X[[self.primary_timestamp]])

        X.drop(self.primary_timestamp, axis=1, inplace=True)
        X = pd.concat([X, time_fea], axis=1)

        self.sfm_ = SelectFromModel(
            lgb.LGBMRegressor(importance_type="gain", random_state=0), threshold=1e-06
        )

        self.sfm_.fit(X, y)

        X = X.iloc[:, self.sfm_.get_support()]

        # lightgbm model use parse time feature
        self.lgb_model.fit(X, y)

        print(f"Feature importance: {self.lgb_model.score()}")

        print("Finish train\n")

        next_step = "predict"
        return next_step

    def predict(self, new_history, pred_record, time_info):
        if self.n_predict % 100 == 0:
            print(f"\nPredict time budget: {time_info['predict']}s")
        self.n_predict += 1

        # type adapter
        pred_record = self.type_adapter.transform(pred_record)

        # Clip numerical features
        if len(self.dtype_cols["num"]) > 0:
            pred_record[self.dtype_cols["num"]] = self.clipped_features.transform(
                pred_record[self.dtype_cols["num"]]
            )

        # parse time feature
        pred_record[self.primary_timestamp] = pd.to_datetime(
            pred_record[self.primary_timestamp], unit="s"
        )
        time_fea = self.calendar_features.transform(
            pred_record[[self.primary_timestamp]]
        )

        pred_record.drop(self.primary_timestamp, axis=1, inplace=True)
        pred_record = pd.concat([pred_record, time_fea], axis=1)

        pred_record = pred_record.iloc[:, self.sfm_.get_support()]

        predictions = self.lgb_model.predict(pred_record)

        if self.n_predict > self.update_interval:
            next_step = "update"
            self.n_predict = 0
        else:
            next_step = "predict"

        return list(predictions), next_step

    def update(self, train_data, test_history_data, time_info):
        print(f"\nUpdate time budget: {time_info['update']}s")

        total_data = pd.concat([train_data, test_history_data])

        self.train(total_data, time_info)

        print("Finish update\n")

        next_step = "predict"
        return next_step

    def save(self, model_dir, time_info):
        print(f"\nSave time budget: {time_info['save']}s")

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

        print("Finish save\n")

    def load(self, model_dir, time_info):
        print(f"\nLoad time budget: {time_info['load']}s")

        pkl_list = pickle.load(open(os.path.join(model_dir, "pkl_list.pkl"), "rb"))

        for attr in pkl_list:
            setattr(
                self,
                attr,
                pickle.load(open(os.path.join(model_dir, f"{attr}.pkl"), "rb")),
            )

        print("Finish load\n")
