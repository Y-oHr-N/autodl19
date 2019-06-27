from sklearn.utils.estimator_checks import check_estimator

from automllib.under_sampling import ModifiedRandomUnderSampler


def test_modified_random_under_sampler() -> None:
    check_estimator(ModifiedRandomUnderSampler)
