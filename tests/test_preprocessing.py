from sklearn.utils.estimator_checks import check_estimator

from automllib.preprocessing import ArithmeticalFeatures
from automllib.preprocessing import Clip
from automllib.preprocessing import CountEncoder
from automllib.preprocessing import ModifiedStandardScaler
from automllib.preprocessing import RowStatistics


def test_arithmetical_features() -> None:
    check_estimator(ArithmeticalFeatures)


def test_clip() -> None:
    check_estimator(Clip)


def test_count_encoder() -> None:
    check_estimator(CountEncoder)


def test_row_statistics() -> None:
    check_estimator(RowStatistics)


def test_modified_standard_scaler() -> None:
    check_estimator(ModifiedStandardScaler)
