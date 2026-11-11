"""check() and estimate() should reject the same bad inputs (C-05, W-03, T-10, T-11)."""

import math

import pytest
import torch

from dl_complexity.coding import TwoPartInputs, TwoPartMDL
from dl_complexity.criteria import (
    AIC,
    BIC,
    HQIC,
    WAIC,
    FitStatistics,
    PosteriorLogLikelihood,
)
from dl_complexity.errors import InputError
from dl_complexity.results import RequirementStatus

CRITERIA = [AIC, BIC, HQIC]

# check() can see these without reading tensor values
BAD_METADATA = {
    "k_negative": FitStatistics(-1.0, -1, 10),
    "k_bool": FitStatistics(-1.0, True, 10),
    "k_float": FitStatistics(-1.0, 1.5, 10),
    "k_none": FitStatistics(-1.0, None, 10),
    "n_zero": FitStatistics(-1.0, 1, 0),
    "n_negative": FitStatistics(-1.0, 1, -5),
    "n_bool": FitStatistics(-1.0, 1, True),
    "n_float": FitStatistics(-1.0, 1, 10.0),
    "ll_vector": FitStatistics(torch.tensor([-1.0, -2.0]), 1, 10),
    "ll_integer_tensor": FitStatistics(torch.tensor(-1), 1, 10),
    "ll_bool": FitStatistics(True, 1, 10),
    "ll_list": FitStatistics([-1.0], 1, 10),
}


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda c: c.__name__)
@pytest.mark.parametrize("case", BAD_METADATA, ids=str)
def test_fitted_criteria_reject_bad_metadata_in_check_and_estimate(criterion, case):
    inputs = BAD_METADATA[case]
    with pytest.raises(InputError):
        criterion().check(inputs=inputs)
    with pytest.raises(InputError):
        criterion().estimate(inputs=inputs)


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda c: c.__name__)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_likelihood_is_rejected_at_evaluation(criterion, value):
    # check() does not read values, estimate() does
    for ll in (value, torch.tensor(value, dtype=torch.float64)):
        assert criterion().check(inputs=FitStatistics(ll, 1, 10)).ready
        with pytest.raises(InputError):
            criterion().estimate(inputs=FitStatistics(ll, 1, 10))


@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda c: c.__name__)
def test_valid_metadata_is_ready_in_check(criterion):
    report = criterion().check(inputs=FitStatistics(-10.0, 2, 50))
    assert report.status is RequirementStatus.READY


@pytest.mark.parametrize("n", [1])
def test_hqic_log_log_domain_is_checked_before_evaluation(n):
    inputs = FitStatistics(-5.0, 1, n)
    with pytest.raises(InputError, match="HQIC"):
        HQIC().check(inputs=inputs)
    with pytest.raises(InputError, match="HQIC"):
        HQIC().estimate(inputs=inputs)
    # n = 2 is allowed (negative penalty)
    assert HQIC().check(inputs=FitStatistics(-5.0, 1, 2)).ready


@pytest.mark.parametrize("criterion", [BIC, HQIC], ids=lambda c: c.__name__)
def test_missing_n_is_reported_not_raised_by_check(criterion):
    report = criterion().check(inputs=FitStatistics(-5.0, 1))
    assert report.status is RequirementStatus.MISSING_INPUTS
    assert report.missing == ("n",)
    with pytest.raises(InputError):
        criterion().estimate(inputs=FitStatistics(-5.0, 1))


def test_aic_does_not_require_n_in_check():
    assert AIC().check(inputs=FitStatistics(-5.0, 1)).ready


@pytest.mark.parametrize("parameter_count", ["trainable_only", "", -1, True, 1.0, None])
@pytest.mark.parametrize("criterion", CRITERIA, ids=lambda c: c.__name__)
def test_parameter_count_policy_is_validated_at_construction(
    criterion, parameter_count
):
    with pytest.raises(InputError):
        criterion(parameter_count=parameter_count)


# WAIC

BAD_DRAW_MATRICES = {
    "one_draw": torch.zeros(1, 3, dtype=torch.float64),
    "vector": torch.zeros(2, dtype=torch.float64),
    "no_observations": torch.zeros(2, 0, dtype=torch.float64),
    "rank3": torch.zeros(2, 3, 1, dtype=torch.float64),
    "integer": torch.zeros(2, 3, dtype=torch.long),
    "list": [[0.0, 0.0], [0.0, 0.0]],
}


@pytest.mark.parametrize("case", BAD_DRAW_MATRICES, ids=str)
def test_waic_rejects_bad_draw_matrix_in_check_and_estimate(case):
    inputs = PosteriorLogLikelihood(BAD_DRAW_MATRICES[case])
    with pytest.raises(InputError):
        WAIC().check(inputs=inputs)
    with pytest.raises(InputError):
        WAIC().estimate(inputs=inputs)


def test_waic_two_draws_is_the_smallest_valid_matrix():
    ll = torch.tensor([[-1.0, -2.0], [-1.5, -2.5]], dtype=torch.float64)
    assert WAIC().check(inputs=PosteriorLogLikelihood(ll)).ready
    assert math.isfinite(
        float(WAIC().estimate(inputs=PosteriorLogLikelihood(ll)).primary.value)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_waic_nonfinite_values_are_rejected_at_evaluation(value):
    ll = torch.tensor([[-1.0, value], [-1.0, -2.0]], dtype=torch.float64)
    assert WAIC().check(inputs=PosteriorLogLikelihood(ll)).ready  # metadata only
    with pytest.raises(InputError):
        WAIC().estimate(inputs=PosteriorLogLikelihood(ll))


# two-part code


@pytest.mark.parametrize("unit", ["kbits", "bit", "nat", "BITS", "", None, 2])
def test_two_part_rejects_unknown_units(unit):
    with pytest.raises(InputError):
        TwoPartMDL(unit=unit)


@pytest.mark.parametrize("unit,scale", [("nats", 1.0), ("bits", 1 / math.log(2))])
@pytest.mark.parametrize("n", [1, 7, 1000])
def test_two_part_per_observation_is_total_divided_by_n(unit, scale, n):
    model_bits, data_nll = 12.0, 30.0
    result = TwoPartMDL(unit=unit).estimate(
        inputs=TwoPartInputs(model_bits=model_bits, data_nll=data_nll, n=n)
    )
    expected_total = (model_bits * math.log(2) + data_nll) * scale
    total = result.quantities["total"]
    per_obs = result.quantities["per_observation"]
    assert per_obs.unit == unit and per_obs.normalization == "per_observation"
    assert math.isclose(float(total.value), expected_total, rel_tol=1e-12)
    assert math.isclose(float(per_obs.value), expected_total / n, rel_tol=1e-12)
    assert math.isclose(float(per_obs.value) * n, float(total.value), rel_tol=1e-12)
