"""WAIC invariants and logsumexp stability (W-04, W-05)."""

import math

import pytest
import torch

from dl_complexity.criteria import WAIC, PosteriorLogLikelihood

DT = torch.float64


def waic(ll):
    q = WAIC().estimate(inputs=PosteriorLogLikelihood(ll)).quantities
    return {name: q[name].value for name in ("waic", "lppd", "p_waic", "elpd_waic")}


def assert_same(a, b, **tol):
    for name in a:
        torch.testing.assert_close(a[name], b[name], **tol, msg=name)


@pytest.fixture
def draws():
    g = torch.Generator().manual_seed(0)
    # columns with different means and spreads
    scale = torch.linspace(0.05, 2.0, 11, dtype=DT)
    center = -torch.linspace(0.1, 5.0, 11, dtype=DT)
    return center + scale * torch.randn(40, 11, generator=g, dtype=DT)


def test_permuting_draws_or_observations_does_not_change_waic(draws):
    reference = waic(draws)
    g = torch.Generator().manual_seed(1)
    rows = torch.randperm(draws.shape[0], generator=g)
    cols = torch.randperm(draws.shape[1], generator=g)
    assert_same(waic(draws[rows]), reference, rtol=1e-12, atol=1e-12)
    assert_same(waic(draws[:, cols]), reference, rtol=1e-12, atol=1e-12)
    assert_same(waic(draws[rows][:, cols]), reference, rtol=1e-12, atol=1e-12)


def test_waic_is_additive_over_observations(draws):
    left, right = draws[:, :4], draws[:, 4:]
    whole, a, b = waic(draws), waic(left), waic(right)
    for name in whole:
        torch.testing.assert_close(
            whole[name], a[name] + b[name], rtol=1e-12, atol=1e-12
        )


def test_per_observation_constant_shifts_lppd_but_not_p_waic(draws):
    shift = torch.linspace(-3.0, 7.0, draws.shape[1], dtype=DT)
    base, moved = waic(draws), waic(draws + shift)  # same constant for every draw
    torch.testing.assert_close(moved["lppd"], base["lppd"] + shift.sum())
    torch.testing.assert_close(moved["p_waic"], base["p_waic"])
    torch.testing.assert_close(moved["waic"], base["waic"] - 2 * shift.sum())


def test_duplicating_draws_fixes_the_sample_variance_convention(draws):
    """lppd stays the same, ddof=1 variance gets multiplied by 2(S-1)/(2S-1)."""
    s = draws.shape[0]
    base, doubled = waic(draws), waic(torch.cat([draws, draws]))
    torch.testing.assert_close(doubled["lppd"], base["lppd"])
    torch.testing.assert_close(
        doubled["p_waic"], base["p_waic"] * 2 * (s - 1) / (2 * s - 1)
    )


def test_identical_draws_have_zero_optimism(draws):
    one = draws[:1].expand(5, -1).clone()
    result = waic(one)
    assert float(result["p_waic"]) == 0.0
    torch.testing.assert_close(result["lppd"], draws[0].sum())


def test_logsumexp_with_distinct_large_negative_values():
    # in column 1 exp(ll) underflows to 0
    ll = torch.tensor([[-1e4, -800.0], [-1e4 - 50, -801.0], [-1e4, -802.0]], dtype=DT)
    assert float(torch.exp(ll[:, 1]).sum()) == 0.0  # the naive route would give -inf
    lppd_0 = -1e4 + math.log((2 + math.exp(-50)) / 3)
    lppd_1 = -800.0 + math.log((1 + math.exp(-1) + math.exp(-2)) / 3)
    var_0 = torch.tensor([-1e4, -1e4 - 50, -1e4], dtype=DT).var(correction=1)
    expected_p = float(var_0) + 1.0  # var(-800, -801, -802) = 1
    result = waic(ll)
    torch.testing.assert_close(result["lppd"], torch.tensor(lppd_0 + lppd_1, dtype=DT))
    torch.testing.assert_close(result["p_waic"], torch.tensor(expected_p, dtype=DT))
    assert math.isclose(float(var_0), 2500 / 3, rel_tol=1e-12)
    torch.testing.assert_close(
        result["waic"], torch.tensor(-2 * (lppd_0 + lppd_1 - expected_p), dtype=DT)
    )


def test_float32_input_keeps_dtype_and_agrees_with_float64(draws):
    low = waic(draws.float())
    high = waic(draws)
    for name in low:
        assert low[name].dtype == torch.float32
        torch.testing.assert_close(low[name].double(), high[name], rtol=1e-5, atol=1e-4)
