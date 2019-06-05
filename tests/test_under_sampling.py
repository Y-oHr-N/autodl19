from sklearn.utils.estimator_checks import check_estimator

from automllib.under_sampling import RandomUnderSampler


def test_random_under_sampler():
    check_estimator(RandomUnderSampler)
