import logging

import pandas as pd

from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .compose import make_model
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = make_model()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        random_state=0
    )

    fit_params = {
        'search_cv__early_stopping_rounds': 10,
        'search_cv__eval_set': [(X_valid, y_valid)],
        'search_cv__verbose': False
    }

    model.fit(X_train, y_train, **fit_params)

    best_score = model.named_steps['search_cv'].best_score_

    logger.info(f'The best score is {best_score:.3f}')

    return model
