"""Exact Gaussian and small arithmetic references, not shape-only checks."""

import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from dl_complexity.bayesian import (
    KFACFactors,
    KFACLaplace,
    KroneckerBlock,
    LaplaceTerms,
    TemperedLikelihood,
    VariationalCode,
    VariationalTerms,
    WBIC,
    prepare_gaussian_linear_kfac,
    prepare_variational_terms,
)
from dl_complexity.errors import InputError


def test_wbic_mean_nll_has_no_extra_beta_or_two():
    n = 100
    result = WBIC().estimate(
        inputs=TemperedLikelihood(torch.tensor([-2.0, -4.0, -6.0]), n, 1 / math.log(n))
    )
    assert result.primary.value == 4
    assert result.primary.unit == "nats"


@pytest.mark.parametrize("n,beta", [(1, 1.0), (100, 1.0), (10, float("nan"))])
def test_wbic_rejects_wrong_temperature(n, beta):
    with pytest.raises(InputError):
        WBIC().estimate(inputs=TemperedLikelihood(torch.tensor([-2.0, -4.0]), n, beta))


def test_wbic_beta_bernoulli_tempered_expectation():
    # Analytic expectation under Beta(a+beta*s, b+beta*(n-s)); samples are explicit preparation.
    n, successes, a, b = 10, 7, 2.0, 3.0
    beta = 1 / math.log(n)
    alpha = torch.tensor(a + beta * successes, dtype=torch.float64)
    beta_posterior = torch.tensor(b + beta * (n - successes), dtype=torch.float64)
    with torch.random.fork_rng():
        torch.manual_seed(101)
        draws = torch.distributions.Beta(alpha, beta_posterior).sample((100000,))
    ll = successes * draws.log() + (n - successes) * torch.log1p(-draws)
    expected = -successes * (
        torch.digamma(alpha) - torch.digamma(alpha + beta_posterior)
    )
    expected -= (n - successes) * (
        torch.digamma(beta_posterior) - torch.digamma(alpha + beta_posterior)
    )
    result = WBIC().estimate(inputs=TemperedLikelihood(ll, n, beta))
    torch.testing.assert_close(result.primary.value, expected, rtol=0, atol=0.03)


def test_variational_terms_and_unit_conversion_preserve_gradients():
    expected_nll = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    kl = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    result = VariationalCode().estimate(inputs=VariationalTerms(expected_nll, kl, n=5))
    torch.testing.assert_close(result.primary.value, expected_nll.new_tensor(6.0))
    torch.testing.assert_close(
        result.primary.to("bits").value, expected_nll.new_tensor(6 / math.log(2))
    )
    result.primary.value.backward()
    assert expected_nll.grad == 1 and kl.grad == 1


def test_variational_exact_posterior_equals_evidence():
    # One y~N(theta,sigma²), theta~N(0,tau²), exact conjugate Gaussian q.
    y, sigma2, tau2 = 1.7, 0.8, 2.3
    variance = 1 / (1 / tau2 + 1 / sigma2)
    mean = variance * y / sigma2
    nll = (
        0.5 * math.log(2 * math.pi * sigma2)
        + 0.5 * ((y - mean) ** 2 + variance) / sigma2
    )
    kl = 0.5 * ((variance + mean**2) / tau2 - 1 + math.log(tau2 / variance))
    reference = 0.5 * math.log(2 * math.pi * (sigma2 + tau2)) + 0.5 * y * y / (
        sigma2 + tau2
    )
    result = VariationalCode().estimate(inputs=VariationalTerms(nll, kl))
    torch.testing.assert_close(
        result.primary.value, torch.tensor(reference, dtype=torch.float64)
    )


def test_explicit_variational_preparer_analytic_kl_and_reparameterisation():
    loc = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    q = Normal(loc, torch.tensor(0.7, dtype=torch.float64))
    prior = Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        terms = prepare_variational_terms(
            q, prior, lambda theta: 0.5 * theta.square(), n_samples=40000
        )
    reference_kl = 0.5 * (0.7**2 + 0.4**2 - 1 - math.log(0.7**2))
    torch.testing.assert_close(terms.kl_divergence, loc.new_tensor(reference_kl))
    torch.testing.assert_close(
        terms.expected_nll, loc.new_tensor(0.5 * (0.7**2 + 0.4**2)), atol=0.01, rtol=0
    )
    VariationalCode().estimate(inputs=terms).primary.value.backward()
    assert torch.isfinite(loc.grad)


def test_negative_analytic_kl_is_rejected_not_clipped():
    with pytest.raises(InputError):
        VariationalCode().estimate(inputs=VariationalTerms(1.0, -0.1))
    # A noisy Monte Carlo estimate may be negative; its provenance is explicit.
    assert (
        VariationalCode()
        .estimate(inputs=VariationalTerms(1.0, -0.1, kl_estimator="monte_carlo"))
        .primary.value
        == 0.9
    )


def test_laplace_supplied_terms_dimension_and_constants():
    result = KFACLaplace().estimate(inputs=LaplaceTerms(-3.0, -2.0, 2, math.log(12)))
    reference = 5 - math.log(2 * math.pi) + 0.5 * math.log(12)
    torch.testing.assert_close(
        result.primary.value, torch.tensor(reference, dtype=torch.float64)
    )


def test_kfac_exact_additive_prior_matches_dense_logdet():
    a = torch.tensor([[2.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    g = torch.tensor([[1.5, 0.2], [0.2, 0.8]], dtype=torch.float64)
    prior, scale = 0.3, 7.0
    precision = scale * torch.kron(g, a) + prior * torch.eye(4, dtype=torch.float64)
    logdet = torch.linalg.slogdet(precision).logabsdet
    factors = KFACFactors(
        -10.0, -3.0, (KroneckerBlock(a, g, scale),), prior, "supplied_gaussian_ggn"
    )
    computed = KFACLaplace().estimate(inputs=factors)
    reference = KFACLaplace().estimate(inputs=LaplaceTerms(-10.0, -3.0, 4, logdet))
    torch.testing.assert_close(computed.primary.value, reference.primary.value)


def test_gaussian_linear_map_exact_evidence_and_state_unchanged():
    x = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
    y = torch.tensor([[0.1], [0.4], [1.7]], dtype=torch.float64)
    variance, alpha = 0.6, 0.9
    model = nn.Linear(1, 1, bias=True).to(dtype=torch.float64)
    augmented = torch.cat([x, torch.ones_like(x)], dim=1)
    precision = augmented.mT @ augmented / variance + alpha * torch.eye(
        2, dtype=torch.float64
    )
    mean = torch.linalg.solve(precision, augmented.mT @ y / variance)
    with torch.no_grad():
        model.weight.copy_(mean[0:1].mT)
        model.bias.copy_(mean[1])
    state = {name: t.clone() for name, t in model.state_dict().items()}
    factors = prepare_gaussian_linear_kfac(
        model, x, y, noise_std=math.sqrt(variance), prior_precision=alpha
    )
    result = KFACLaplace().estimate(inputs=factors)
    # Integrate theta analytically: y ~ N(0, sigma² I + X X.T / alpha).
    covariance = (
        variance * torch.eye(3, dtype=torch.float64) + augmented @ augmented.mT / alpha
    )
    reference = -0.0 + 0.5 * (
        3 * math.log(2 * math.pi)
        + torch.linalg.slogdet(covariance).logabsdet
        + (y.mT @ torch.linalg.solve(covariance, y)).squeeze()
    )
    torch.testing.assert_close(result.primary.value, reference)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name])


@pytest.mark.parametrize(
    "factor",
    [
        torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float64),
        torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64)),
    ],
)
def test_kfac_rejects_invalid_psd_factors(factor):
    inputs = KFACFactors(
        -1.0,
        -1.0,
        (KroneckerBlock(factor, torch.ones((1, 1), dtype=torch.float64)),),
        1.0,
        "declared",
    )
    with pytest.raises(InputError):
        KFACLaplace().estimate(inputs=inputs)


def test_python_constants_do_not_promote_float32_objective():
    nll = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
    result = VariationalCode().estimate(inputs=VariationalTerms(nll, 0.5))
    assert result.primary.value.dtype == torch.float32
    result = KFACLaplace().estimate(inputs=LaplaceTerms(-nll, -0.5, 1, 0.1))
    assert result.primary.value.dtype == torch.float32


def test_gaussian_linear_multiple_outputs_with_bias_matches_dense_precision():
    x = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.3]], dtype=torch.float64)
    y = torch.tensor([[0.1, 0.2], [0.3, -0.4], [0.8, 0.9]], dtype=torch.float64)
    model = nn.Linear(2, 2, bias=True).to(dtype=torch.float64)
    factors = prepare_gaussian_linear_kfac(
        model, x, y, noise_std=0.7, prior_precision=0.4
    )
    block = factors.blocks[0]
    dense = block.scale * torch.kron(block.output_factor, block.activation_factor)
    dense += 0.4 * torch.eye(6, dtype=torch.float64)
    reference = LaplaceTerms(
        factors.log_likelihood,
        factors.log_prior,
        6,
        torch.linalg.slogdet(dense).logabsdet,
    )
    torch.testing.assert_close(
        KFACLaplace().estimate(inputs=factors).primary.value,
        KFACLaplace().estimate(inputs=reference).primary.value,
    )


@pytest.mark.parametrize("q_is_joint", [False, True])
def test_mc_kl_reduces_joint_density_before_subtraction(q_is_joint):
    # Normal's batch axis and Independent's event axis represent the same joint
    # law here. Broadcasting their raw log_prob values would double-count p or q.
    components = Normal(
        torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    )
    joint = torch.distributions.Independent(components, 1)
    q, prior = (joint, components) if q_is_joint else (components, joint)
    with torch.random.fork_rng():
        torch.manual_seed(113)
        terms = prepare_variational_terms(
            q, prior, lambda theta: theta.square().sum(), n_samples=8, analytic_kl=False
        )
    torch.testing.assert_close(
        terms.kl_divergence, torch.zeros((), dtype=torch.float64), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "terms",
    [
        VariationalTerms(1.0, 1.0, n=0),
        VariationalTerms(torch.ones(2), 1.0),
        VariationalTerms(1.0, torch.tensor(1)),
        VariationalTerms(1.0, 1.0, kl_estimator="unknown"),
    ],
)
def test_variational_preflight_rejects_invalid_metadata(terms):
    with pytest.raises(InputError):
        VariationalCode().check(inputs=terms)


def test_wbic_preflight_rejects_integer_log_likelihood():
    with pytest.raises(InputError):
        WBIC().check(
            inputs=TemperedLikelihood(torch.tensor([-1, -2]), 100, 1 / math.log(100))
        )


@pytest.mark.parametrize(
    "inputs",
    [
        KFACFactors(-1.0, -2.0, (), 1.0, "ggn"),
        LaplaceTerms(-1.0, -2.0, 0, 1.0),
        LaplaceTerms(-1.0, -2.0, 1, torch.ones(2)),
        KFACFactors(
            -1.0,
            -2.0,
            (KroneckerBlock(torch.ones((2, 3)), torch.ones((1, 1))),),
            1.0,
            "ggn",
        ),
    ],
)
def test_laplace_preflight_rejects_invalid_metadata(inputs):
    with pytest.raises(InputError):
        KFACLaplace().check(inputs=inputs)


def test_laplace_preflight_does_not_factorise_or_scan_tensors(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Metadata preflight performed a tensor scan or eigendecomposition")

    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    factors = KFACFactors(
        -1.0, -2.0, (KroneckerBlock(torch.eye(2), torch.eye(1)),), 1.0, "ggn"
    )
    assert KFACLaplace().check(inputs=factors).ready
