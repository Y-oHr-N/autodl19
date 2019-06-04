from sklearn.utils.estimator_checks import check_estimator

from automllib.preprocessing import Clip
from automllib.preprocessing import CountEncoder
from automllib.preprocessing import RowStatistics
from automllib.preprocessing import ModifiedStandardScaler
from automllib.preprocessing import SubtractedFeatures


def test_clip():
    check_estimator(Clip)

def test_count_encoder():
    check_estimator(CountEncoder)

def test_row_statistics():
    check_estimator(RowStatistics)

def test_modified_standard_scaler():
    check_estimator(ModifiedStandardScaler)

def test_subtracted_features():
    check_estimator(SubtractedFeatures)
