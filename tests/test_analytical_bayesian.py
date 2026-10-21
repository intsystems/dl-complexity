"""Conjugate-model oracles for the public Bayesian estimators.

The reference evidence integrates parameters in observation space, rather than
repeating the KFAC precision calculation. WBIC sigma points are deterministic
quadrature for a quadratic integrand, not posterior samples or a sampler test.
"""

import math

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from dl_complexity.bayesian import (
    KFACLaplace,
    TemperedLikelihood,
    VariationalCode,
    VariationalTerms,
    WBIC,
    prepare_gaussian_linear_kfac,
    prepare_variational_terms,
)


LINEAR_CASES = {
    "normal_mean": {
        "x": [[1.0], [1.0], [1.0], [1.0], [1.0]],
        "y": [[-0.5], [0.0], [0.5], [1.0], [1.5]],
        "bias": False,
        "sigma2": 0.75,
        "alpha": 0.6,
    },
    "multioutput_intercept": {
        "x": [[-1.0, 0.5], [0.0, 1.0], [0.5, -1.0], [2.0, 0.25]],
        "y": [[0.0, -0.5], [0.5, 1.0], [1.0, 0.25], [1.5, 2.0]],
        "bias": True,
        "sigma2": 0.5,
        "alpha": 1.25,
    },
    "underdetermined": {
        "x": [[1.0, -0.5, 2.0], [0.0, 1.0, -1.0]],
        "y": [[0.75], [-0.5]],
        "bias": False,
        "sigma2": 1.5,
        "alpha": 0.25,
    },
}


def linear_problem(case_name, dtype):
    """Construct an exact ridge MAP, with a proper prior even when n < d."""
    case = LINEAR_CASES[case_name]
    x = torch.tensor(case["x"], dtype=dtype)
    y = torch.tensor(case["y"], dtype=dtype)
    design = x.to(torch.float64)
    if case["bias"]:
        design = torch.cat([design, torch.ones((len(x), 1), dtype=design.dtype)], 1)
    precision = design.mT @ design / case["sigma2"]
    precision += case["alpha"] * torch.eye(design.shape[1], dtype=design.dtype)
    mean = torch.linalg.solve(precision, design.mT @ y.double() / case["sigma2"])
    # Model construction is not part of the estimator and must not leak RNG state.
    with torch.random.fork_rng():
        model = nn.Linear(x.shape[1], y.shape[1], bias=case["bias"], dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(mean[: x.shape[1]].mT)
        if model.bias is not None:
            model.bias.copy_(mean[-1])
    return case, x, y, design, precision, mean, model


def integrated_gaussian_nll(design, targets, sigma2, alpha):
    """Integrate theta: each output y ~ N(0, sigma2*I + X*X.T/alpha)."""
    targets = targets.double()
    covariance = sigma2 * torch.eye(design.shape[0], dtype=torch.float64)
    covariance += design @ design.mT / alpha
    cholesky = torch.linalg.cholesky(covariance)
    logdet = 2 * cholesky.diag().log().sum()
    quadratic = (targets * torch.cholesky_solve(targets, cholesky)).sum()
    n, outputs = targets.shape
    return 0.5 * (
        outputs * (n * math.log(2 * math.pi) + logdet.item()) + quadratic.item()
    )


@pytest.mark.parametrize("case_name", list(LINEAR_CASES))
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_kfac_map_matches_integrated_linear_gaussian(case_name, dtype, record_analytic):
    """KFAC is exact for this one-layer Gaussian model, including constants."""
    case, x, y, design, precision, mean, model = linear_problem(case_name, dtype)
    original_state = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }
    model.train()
    factors = prepare_gaussian_linear_kfac(
        model,
        x,
        y,
        noise_std=math.sqrt(case["sigma2"]),
        prior_precision=case["alpha"],
    )
    result = KFACLaplace().estimate(inputs=factors)
    expected = integrated_gaussian_nll(design, y, case["sigma2"], case["alpha"])
    tolerance = 5e-6 if dtype == torch.float32 else 1e-10
    dtype_name = str(dtype).split(".")[-1]
    prefix = f"kfac_{case_name}_{dtype_name}"
    details = {
        "n": len(x),
        "input_features": x.shape[1],
        "outputs": y.shape[1],
        "bias": case["bias"],
        "sigma2_known": case["sigma2"],
        "prior_precision": case["alpha"],
        "dtype": dtype_name,
        "map_solved_analytically": True,
        "seed": None,
    }
    record_analytic(
        case_id=prefix + "_evidence",
        method="KFACLaplace",
        model=case_name,
        expected=expected,
        actual=result.primary.value,
        unit="nats",
        oracle="-log N(Y; 0, sigma^2 I + X_aug X_aug.T / alpha), independent outputs",
        atol=tolerance,
        rtol=tolerance,
        details=details,
    )
    # These component oracles catch missing density constants independently of Z.
    residual = y.double() - design @ mean
    nll = 0.5 * (
        y.numel() * math.log(2 * math.pi * case["sigma2"])
        + residual.square().sum().item() / case["sigma2"]
    )
    negative_log_prior = 0.5 * (
        mean.numel() * math.log(2 * math.pi / case["alpha"])
        + case["alpha"] * mean.square().sum().item()
    )
    half_logdet = (
        y.shape[1] * torch.linalg.cholesky(precision).diag().log().sum().item()
    )
    for name, expected_component in (
        ("negative_log_likelihood", nll),
        ("negative_log_prior", negative_log_prior),
        ("half_logdet_precision", half_logdet),
    ):
        record_analytic(
            case_id=prefix + "_" + name,
            method="KFACLaplace." + name,
            model=case_name,
            expected=expected_component,
            actual=result.quantities[name].value,
            unit="nats",
            oracle="Normalized Gaussian log density; precision logdet from dense Cholesky",
            atol=tolerance,
            rtol=tolerance,
            details=details,
        )
    assert result.primary.unit == "nats"
    assert result.config["dimension"] == mean.numel()
    assert result.primary.value.dtype == dtype
    assert model.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)


def test_laplace_off_map_has_known_quadratic_error(record_analytic):
    """Readiness cannot establish MAP: displacement produces an exact known gap."""
    case, x, y, design, precision, mean, model = linear_problem(
        "multioutput_intercept", torch.float64
    )
    displacement = torch.tensor(
        [[0.25, -0.5], [0.5, 0.0], [-0.25, 0.25]], dtype=torch.float64
    )
    with torch.no_grad():
        model.weight.copy_((mean + displacement)[:-1].mT)
        model.bias.copy_((mean + displacement)[-1])
    factors = prepare_gaussian_linear_kfac(
        model,
        x,
        y,
        noise_std=math.sqrt(case["sigma2"]),
        prior_precision=case["alpha"],
    )
    assert KFACLaplace().check(inputs=factors).ready
    value = KFACLaplace().estimate(inputs=factors).primary.value
    evidence = integrated_gaussian_nll(design, y, case["sigma2"], case["alpha"])
    gap = 0.5 * (displacement * (precision @ displacement)).sum().item()
    assert gap > 0
    record_analytic(
        case_id="laplace_off_map_gap",
        method="KFACLaplace.off_map_gap",
        model="multioutput_intercept",
        expected=gap,
        actual=value - evidence,
        unit="nats",
        oracle="Off-MAP Laplace expression minus true -log Z = tr(delta.T H delta)/2",
        details={"map_assumption_intentionally_false": True, "ready_is_not_map": True},
    )


def normal_mean_reference():
    """Return a conjugate Normal mean problem and its exact posterior moments."""
    case = LINEAR_CASES["normal_mean"]
    y = torch.tensor(case["y"], dtype=torch.float64).flatten()
    sigma2, alpha = case["sigma2"], case["alpha"]
    posterior_variance = 1 / (alpha + len(y) / sigma2)
    posterior_mean = posterior_variance * y.sum().item() / sigma2
    evidence = integrated_gaussian_nll(
        torch.ones((len(y), 1), dtype=torch.float64), y[:, None], sigma2, alpha
    )
    return y, sigma2, alpha, posterior_mean, posterior_variance, evidence


def gaussian_expected_nll(y, sigma2, mean, variance):
    """Use E[(y-theta)^2] = (y-E theta)^2 + Var(theta), not sample averages."""
    return 0.5 * (
        len(y) * math.log(2 * math.pi * sigma2)
        + ((y - mean).square().sum().item() + len(y) * variance) / sigma2
    )


@pytest.mark.parametrize("q_kind", ["exact_posterior", "prior", "shifted_posterior"])
def test_variational_identity_has_exact_gaussian_gap(q_kind, record_analytic):
    """Negative ELBO equals -log evidence + KL(q || exact posterior)."""
    y, sigma2, alpha, posterior_mean, posterior_variance, evidence = (
        normal_mean_reference()
    )
    mean, variance = posterior_mean, posterior_variance
    if q_kind == "prior":
        mean, variance = 0.0, 1 / alpha
    elif q_kind == "shifted_posterior":
        mean, variance = posterior_mean + 0.4, posterior_variance * 1.5
    nll = gaussian_expected_nll(y, sigma2, mean, variance)
    kl_prior = 0.5 * (alpha * (variance + mean**2) - 1 - math.log(alpha * variance))
    kl_posterior = 0.5 * (
        (variance + (mean - posterior_mean) ** 2) / posterior_variance
        - 1
        + math.log(posterior_variance / variance)
    )
    result = VariationalCode().estimate(
        inputs=VariationalTerms(nll, kl_prior, n=len(y), kl_estimator="analytic")
    )
    record_analytic(
        case_id="variational_" + q_kind + "_free_energy",
        method="VariationalCode",
        model="normal_mean",
        expected=evidence + kl_posterior,
        actual=result.primary.value,
        unit="nats",
        oracle="F(q) = -log Z + KL(N(m_q,v_q) || N(m_post,v_post))",
        details={"q": q_kind, "q_mean": mean, "q_variance": variance},
    )
    record_analytic(
        case_id="variational_" + q_kind + "_gap",
        method="VariationalCode.posterior_gap",
        model="normal_mean",
        expected=kl_posterior,
        actual=result.primary.value - evidence,
        unit="nats",
        oracle="KL(q||posterior), zero only for q equal to the conjugate posterior",
        details={"q": q_kind},
    )
    assert kl_posterior >= 0
    assert result.primary.value.item() >= evidence - 1e-10
    torch.testing.assert_close(
        result.primary.to("bits").value,
        result.primary.value / math.log(2),
        rtol=0,
        atol=0,
    )
    normalized = result.primary.per_observation(len(y))
    assert normalized.normalization == "per_observation"
    assert normalized.value.item() == result.primary.value.item() / len(y)


@pytest.mark.parametrize("q_kind", ["exact_posterior", "shifted_posterior"])
def test_variational_preparation_matches_analytic_moments(q_kind, record_analytic):
    """The explicit preparer is MC: compare with an a priori eight-SE tolerance."""
    y, sigma2, alpha, posterior_mean, posterior_variance, evidence = (
        normal_mean_reference()
    )
    mean, variance = posterior_mean, posterior_variance
    if q_kind == "shifted_posterior":
        mean, variance = posterior_mean + 0.4, posterior_variance * 1.5
    loc = torch.tensor(mean, dtype=torch.float64, requires_grad=True)
    q = Normal(loc, loc.new_tensor(math.sqrt(variance)))
    prior = Normal(loc.new_tensor(0), loc.new_tensor(1 / math.sqrt(alpha)))
    n_samples, seed = 4096, 281

    def total_nll(theta):
        """Normalized total-data likelihood, without prior or minibatch scaling."""
        return 0.5 * (
            len(y) * math.log(2 * math.pi * sigma2)
            + (y - theta).square().sum() / sigma2
        )

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        terms = prepare_variational_terms(
            q, prior, total_nll, n_samples=n_samples, n=len(y), analytic_kl=True
        )
    expected_nll = gaussian_expected_nll(y, sigma2, mean, variance)
    kl_prior = 0.5 * (alpha * (variance + mean**2) - 1 - math.log(alpha * variance))
    # Var((theta-ybar)^2)=2v^2+4v(m-ybar)^2 for a Normal theta.
    nll_variance = (
        len(y) ** 2
        / (4 * sigma2**2)
        * (2 * variance**2 + 4 * variance * (mean - y.mean().item()) ** 2)
    )
    standard_error = math.sqrt(nll_variance / n_samples)
    details = {
        "q": q_kind,
        "n_samples": n_samples,
        "seed": seed,
        "theoretical_nll_standard_error": standard_error,
        "tolerance_standard_errors": 8,
        "not_exact_quadrature": True,
    }
    record_analytic(
        case_id="variational_mc_" + q_kind + "_nll",
        method="prepare_variational_terms.expected_nll",
        model="normal_mean",
        expected=expected_nll,
        actual=terms.expected_nll,
        unit="nats",
        oracle="Exact Normal moments; MC tolerance = 8 sqrt(Var_q(total NLL)/S)",
        atol=8 * standard_error,
        rtol=0,
        category="approximation",
        details=details,
    )
    record_analytic(
        case_id="variational_mc_" + q_kind + "_kl",
        method="prepare_variational_terms.analytic_kl",
        model="normal_mean",
        expected=kl_prior,
        actual=terms.kl_divergence,
        unit="nats",
        oracle="KL(N(m,v)||N(0,alpha^-1)) = [alpha(v+m^2)-1-log(alpha*v)]/2",
        details=details,
    )
    result = VariationalCode().estimate(inputs=terms)
    record_analytic(
        case_id="variational_mc_" + q_kind + "_free_energy",
        method="VariationalCode.with_mc_preparation",
        model="normal_mean",
        expected=expected_nll + kl_prior,
        actual=result.primary.value,
        unit="nats",
        oracle="Exact expected NLL + analytic KL; only NLL is sampled",
        atol=8 * standard_error,
        rtol=0,
        category="approximation",
        details=details,
    )
    result.primary.value.backward()
    assert torch.isfinite(loc.grad)


@pytest.mark.parametrize("n", [2, 5, 20])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_wbic_normal_tempered_quadrature(n, dtype, record_analytic):
    """Two moment-matched nodes integrate quadratic NLL exactly, without MCMC."""
    sigma2, alpha = 0.75, 0.6
    # A fixed dyadic sequence avoids random data generation and dtype drift.
    observations = [-0.5, 0.0, 0.5, 1.0, 1.5] * 4
    y = torch.tensor(observations[:n], dtype=torch.float64)
    beta = 1 / math.log(n)
    variance = 1 / (alpha + beta * n / sigma2)
    mean = variance * beta * y.sum().item() / sigma2
    nodes = torch.tensor(
        [mean - math.sqrt(variance), mean + math.sqrt(variance)], dtype=dtype
    )
    total_log_likelihood = -0.5 * (
        n * math.log(2 * math.pi * sigma2)
        + (y.to(dtype)[None, :] - nodes[:, None]).square().sum(1) / sigma2
    )
    result = WBIC().estimate(
        inputs=TemperedLikelihood(total_log_likelihood, n=n, beta=beta)
    )
    expected = gaussian_expected_nll(y, sigma2, mean, variance)
    evidence = integrated_gaussian_nll(
        torch.ones((n, 1), dtype=torch.float64), y[:, None], sigma2, alpha
    )
    tolerance = 5e-6 if dtype == torch.float32 else 1e-10
    record_analytic(
        case_id=f"wbic_normal_n{n}_{str(dtype).split('.')[-1]}",
        method="WBIC",
        model="normal_mean_tempered",
        expected=expected,
        actual=result.primary.value,
        unit="nats",
        oracle="E_beta[NLL] from m_beta,v_beta; two equally weighted sigma points are exact for a quadratic",
        atol=tolerance,
        rtol=tolerance,
        details={
            "n": n,
            "beta": beta,
            "tempered_mean": mean,
            "tempered_variance": variance,
            "quadrature_not_samples": True,
            "does_not_test_sampler": True,
            "exact_finite_n_negative_log_evidence": evidence,
            "wbic_minus_exact_evidence": expected - evidence,
            "sigma2_known": sigma2,
            "prior_precision": alpha,
        },
    )
    assert not math.isclose(expected, evidence, abs_tol=1e-4)
    assert result.primary.value.dtype == dtype
    assert result.primary.unit == "nats"
