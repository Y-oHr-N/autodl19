from sklearn.utils.estimator_checks import check_estimator

from automllib.feature_selection import DropCollinearFeatures
from automllib.feature_selection import DropDriftFeatures
from automllib.feature_selection import FrequencyThreshold
from automllib.feature_selection import NAProportionThreshold


def test_drop_colinear_features() -> None:
    check_estimator(DropCollinearFeatures)


def test_drop_drift_features() -> None:
    check_estimator(DropDriftFeatures)


def test_frequency_threshold() -> None:
    check_estimator(FrequencyThreshold)


def test_na_proportion_threshold() -> None:
    check_estimator(NAProportionThreshold)
