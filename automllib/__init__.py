import logging

import colorlog

from . import automl  # noqa
from . import base  # noqa
from . import compose  # noqa
from . import feature_extraction  # noqa
from . import feature_selection  # noqa
from . import impute  # noqa
from . import model_selection  # noqa
from . import preprocessing  # noqa
from . import table_join  # noqa
from . import under_sampling  # noqa
from . import utils  # noqa

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s'
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
