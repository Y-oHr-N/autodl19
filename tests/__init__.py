import logging

import colorlog

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s'
)

handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

try:
    import comet_ml
except ImportError as e:
    logger.exception(e)
