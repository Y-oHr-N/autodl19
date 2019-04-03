from typing import Any
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin


class Model(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, info: Dict[str, Any]) -> None:
        self.info = info

    def fit(
        self,
        train_data: Dict[str, pd.DataFrame],
        train_label: pd.Series,
        time_remain: float
    ) -> None:
        pass

    def predict(
        self,
        test_data: pd.DataFrame,
        time_remain: float
    ) -> pd.Series:
        n_samples = len(test_data)

        return pd.Series(np.zeros(n_samples))
