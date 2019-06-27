from sklearn.utils.estimator_checks import check_estimator

from automllib.impute import ModifiedSimpleImputer


def test_modified_simple_imputer() -> None:
    check_estimator(ModifiedSimpleImputer)
