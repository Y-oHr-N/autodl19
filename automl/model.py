import os
from typing import Any
from typing import Dict

os.system('pip install numpy pandas')

import numpy as np
import pandas as pd


class Model(object):
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
