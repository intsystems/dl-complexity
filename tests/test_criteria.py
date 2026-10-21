"""Independent small-value oracles for the public criteria interface."""

import math

import pytest
import torch

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


@pytest.mark.parametrize(
    "method,expected",
    [
        (AIC(), 206),
        (BIC(), 200 + 3 * math.log(100)),
        (HQIC(), 200 + 6 * math.log(math.log(100))),
    ],
)
def test_native_classical_oracle(method, expected):
    ll = torch.tensor(-100.0, dtype=torch.float64, requires_grad=True)
    result = method.estimate(inputs=FitStatistics(ll, k=3, n=100))
    torch.testing.assert_close(result.primary.value, ll.new_tensor(expected))
    assert result.kind == "information_criterion"
    assert result.primary.unit == "native_ic"
    assert result.primary.value.dtype == torch.float64
    result.primary.value.backward()
    torch.testing.assert_close(ll.grad, ll.new_tensor(-2))
    with pytest.raises(InputError):
        result.primary.to("bits")
    torch.testing.assert_close(
        result.primary.per_observation(100).value, ll.new_tensor(expected / 100)
    )


def test_aic_without_n_and_zero_parameters():
    assert AIC().estimate(inputs=FitStatistics(-5.0, k=0)).primary.value == 10
    assert (
        BIC().check(inputs=FitStatistics(-5.0, k=1)).status
        is RequirementStatus.MISSING_INPUTS
    )
    with pytest.raises(InputError):
        BIC().estimate(inputs=FitStatistics(-5.0, k=1))


def test_hqic_small_n_is_not_clipped():
    result = HQIC().estimate(inputs=FitStatistics(-5.0, k=1, n=2))
    assert result.quantities["penalty"].value < 0
    with pytest.raises(InputError):
        HQIC().estimate(inputs=FitStatistics(-5.0, k=1, n=1))


@pytest.mark.parametrize(
    "inputs",
    [
        FitStatistics(float("nan"), 1, 10),
        FitStatistics(-1.0, -1, 10),
        FitStatistics(-1.0, True, 10),
        FitStatistics(-1.0, 1, 0),
    ],
)
def test_bad_fit_inputs_rejected(inputs):
    with pytest.raises(InputError):
        AIC().estimate(inputs=inputs)


def test_waic_analytic_two_by_two():
    # Two equally weighted posterior draws; each observation has values .5/.25.
    ll = torch.tensor(
        [[math.log(0.5), math.log(0.25)], [math.log(0.25), math.log(0.5)]],
        dtype=torch.float64,
        requires_grad=True,
    )
    result = WAIC().estimate(inputs=PosteriorLogLikelihood(ll))
    lppd = 2 * math.log(0.375)
    p_waic = math.log(2) ** 2
    torch.testing.assert_close(
        result.primary.value, ll.new_tensor(-2 * (lppd - p_waic))
    )
    torch.testing.assert_close(result.quantities["lppd"].value, ll.new_tensor(lppd))
    torch.testing.assert_close(result.quantities["p_waic"].value, ll.new_tensor(p_waic))
    assert result.config["variance_ddof"] == 1
    result.primary.value.backward()
    assert bool(torch.isfinite(ll.grad).all())


def test_waic_numerical_stability_and_no_parameter_count():
    ll = torch.full((3, 2), -10000.0, dtype=torch.float64)
    result = WAIC().estimate(inputs=PosteriorLogLikelihood(ll))
    assert result.primary.value == 40000
    assert result.quantities["p_waic"].value == 0


@pytest.mark.parametrize(
    "ll",
    [
        torch.ones(1, 3),
        torch.ones(2),
        torch.ones(2, 0),
        torch.tensor([[float("nan")], [0.0]]),
    ],
)
def test_waic_rejects_invalid_draw_matrix(ll):
    with pytest.raises(InputError):
        WAIC().estimate(inputs=PosteriorLogLikelihood(ll))


@pytest.mark.parametrize(
    "ll", [torch.tensor([-1.0, -2.0]), torch.tensor(-1), [-1.0, -2.0], True]
)
def test_fitted_preflight_rejects_nonscalar_or_nonfloating_likelihood(ll):
    with pytest.raises(InputError):
        AIC().check(inputs=FitStatistics(ll, 1, 10))
