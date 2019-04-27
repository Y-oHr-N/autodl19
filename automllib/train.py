import logging

from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .compose import make_model
from .constants import ONE_DIM_ARRAY_TYPE
from .constants import TWO_DIM_ARRAY_TYPE
from .utils import timeit

logger = logging.getLogger(__name__)


@timeit
def train(X: TWO_DIM_ARRAY_TYPE, y: ONE_DIM_ARRAY_TYPE) -> Pipeline:
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

    try:
        best_score = model.named_steps['search_cv'].best_score_

        logger.info(f'The best score is {best_score:.3f}')

    except:
        logger.warning(f'No trials are completed yet.')

    return model
