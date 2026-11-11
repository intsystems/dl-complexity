"""Checks of the oracles in sampling_oracles.py against exact draws and 2D grids."""

import math

import pytest
import torch

from dl_complexity.bayesian import WBIC, TemperedLikelihood
from dl_complexity.criteria import WAIC, PosteriorLogLikelihood
from sampling_oracles import LinearGaussian, ProductModel, make_regression_data

LOG_2PI = math.log(2 * math.pi)


def _linear_problem(n=60, sigma=0.7, alpha=1.0, seed=1):
    x, y = make_regression_data(n, [0.8, -0.5, 0.0], sigma=sigma, seed=seed)
    return LinearGaussian(x, y, sigma=sigma, alpha=alpha)


def _gaussian_draws(mean, covariance, count, seed):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(count, mean.numel(), generator=generator, dtype=torch.float64)
    return mean + noise @ torch.linalg.cholesky(covariance).T


def _pointwise_log_likelihood(problem, weights):
    residual = problem.y.reshape(1, -1) - weights @ problem.x.T
    return -0.5 * (LOG_2PI + math.log(problem.sigma**2)) - 0.5 * residual**2 / (
        problem.sigma**2
    )


def test_linear_gaussian_waic_oracle_matches_exact_posterior_draws(record_analytic):
    """Closed-form lppd and p_waic agree with exact draws through the WAIC reducer."""
    problem = _linear_problem()
    mean, covariance = problem.posterior(beta=1.0)
    draws = _gaussian_draws(mean, covariance, 200_000, seed=5)
    result = WAIC().estimate(
        inputs=PosteriorLogLikelihood(_pointwise_log_likelihood(problem, draws))
    )
    oracle = problem.waic()
    for name, atol in (("lppd", 0.01), ("p_waic", 0.05), ("waic", 0.12)):
        record_analytic(
            case_id=f"oracle.linear_gaussian.waic.{name}",
            method=f"oracle.WAIC.{name}",
            model="Bayesian linear regression, known noise",
            expected=oracle[name],
            actual=result.quantities[name].value,
            unit="native_ic" if name == "waic" else "nats",
            oracle="E p(y|w) = N(y|mu, sigma^2+s^2); Var log p = (r^2 s^2 + s^4/2)/sigma^4",
            atol=atol,
            rtol=0.0,
            category="approximation",
            details={"draws": 200_000, "n": problem.n, "d": problem.d},
        )


@pytest.mark.parametrize("localization", [0.0, 40.0])
def test_linear_gaussian_wbic_oracle_matches_exact_tempered_draws(
    localization, record_analytic
):
    """E_beta[nL_n] in closed form, with and without localization, vs exact draws."""
    problem = _linear_problem()
    beta = 1 / math.log(problem.n)
    center = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    mean, covariance = problem.posterior(beta, localization, center)
    draws = _gaussian_draws(mean, covariance, 200_000, seed=6)
    totals = _pointwise_log_likelihood(problem, draws).sum(1)
    standard_error = float(totals.std() / math.sqrt(totals.numel()))
    expected = problem.expected_total_nll(beta, localization, center)
    if localization == 0.0:
        actual = (
            WBIC()
            .estimate(inputs=TemperedLikelihood(totals, n=problem.n, beta=beta))
            .primary.value
        )
    else:
        actual = -totals.mean()
    record_analytic(
        case_id=f"oracle.linear_gaussian.expected_nll.gamma{localization:g}",
        method="oracle.E_beta[nL_n]",
        model="Bayesian linear regression, known noise",
        expected=expected,
        actual=actual,
        unit="nats",
        oracle="nL_n(m) + tr(X^T X Sigma) / (2 sigma^2)",
        atol=5 * standard_error,
        rtol=0.0,
        category="approximation",
        details={"beta": beta, "localization": localization, "se": standard_error},
    )


def test_localized_linear_posterior_moves_towards_center():
    """Localization adds gamma*I to the precision and gamma*w0 to the linear term."""
    problem = _linear_problem()
    center = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    plain_mean, plain_cov = problem.posterior(0.25)
    local_mean, local_cov = problem.posterior(0.25, 1e6, center)
    torch.testing.assert_close(local_mean, center, atol=1e-4, rtol=0)
    assert torch.all(local_cov.diag() < plain_cov.diag())
    assert torch.all((local_mean - center).abs() < (plain_mean - center).abs())


def _brute_force_product(problem, beta, half=7.0, size=2_401):
    grid = torch.linspace(-half, half, size, dtype=torch.float64)
    a, b = torch.meshgrid(grid, grid, indexing="ij")
    sxx = float(problem.x @ problem.x)
    sxy = float(problem.x @ problem.y)
    syy = float(problem.y @ problem.y)
    sigma2 = problem.sigma**2
    nll = (
        0.5 * problem.n * (LOG_2PI + math.log(sigma2))
        + 0.5 * (syy - 2 * a * b * sxy + (a * b) ** 2 * sxx) / sigma2
    )
    log_weight = -beta * nll - 0.5 * problem.alpha * (a**2 + b**2)
    weight = torch.exp(log_weight - log_weight.max())
    weight = weight / weight.sum()
    return a, b, nll, weight


@pytest.mark.parametrize("beta", [1.0, 1 / math.log(200)], ids=["beta1", "wbic"])
def test_product_model_quadrature_matches_two_dimensional_grid(beta, record_analytic):
    """Analytic b | a integration plus 1D quadrature equals a dense 2D grid."""
    x, y = make_regression_data(200, [0.0], sigma=1.0, seed=2)
    problem = ProductModel(x, y, sigma=1.0, alpha=1.0)
    a, b, nll, weight = _brute_force_product(problem, beta)
    moments = problem.moments(beta)
    brute = {
        "E[a^2]": float((weight * a**2).sum()),
        "E[b^2]": float((weight * b**2).sum()),
        "E[ab]": float((weight * a * b).sum()),
        "E[(ab)^2]": float((weight * (a * b) ** 2).sum()),
    }
    for name, value in brute.items():
        record_analytic(
            case_id=f"oracle.product.beta{beta:.3f}.{name}",
            method="oracle.ProductModel.moments",
            model="y = a b x + eps, true ab = 0",
            expected=value,
            actual=moments[name],
            unit="dimensionless",
            oracle="b | a Gaussian, trapezoid over a; checked against a 2401^2 grid",
            atol=1e-6,
            rtol=1e-6,
            category="exact",
            details={"beta": beta},
        )
    record_analytic(
        case_id=f"oracle.product.beta{beta:.3f}.expected_nll",
        method="oracle.ProductModel.expected_total_nll",
        model="y = a b x + eps, true ab = 0",
        expected=float((weight * nll).sum()),
        actual=problem.expected_total_nll(beta),
        unit="nats",
        oracle="b | a Gaussian, trapezoid over a; checked against a 2401^2 grid",
        atol=1e-6,
        rtol=1e-9,
        category="exact",
        details={"beta": beta},
    )


def test_product_model_posterior_is_symmetric_in_a_and_b():
    """The likelihood depends on ab and the prior is iid, so E[a^2] = E[b^2] exactly.

    E[a^2] uses only the marginal quadrature weights, whereas E[b^2] uses the
    conditional Gaussian moments, so agreement checks the conditional formulas.
    """
    x, y = make_regression_data(500, [0.3], sigma=0.8, seed=3)
    problem = ProductModel(x, y, sigma=0.8, alpha=2.0)
    for beta in (1.0, 0.3, 1 / math.log(500)):
        moments = problem.moments(beta)
        assert moments["E[a^2]"] == pytest.approx(moments["E[b^2]"], rel=1e-9)


def test_product_model_waic_matches_two_dimensional_grid(record_analytic):
    """Pointwise mixture formulas for lppd and Var log p agree with a dense grid."""
    x, y = make_regression_data(200, [0.0], sigma=1.0, seed=2)
    problem = ProductModel(x, y, sigma=1.0, alpha=1.0, grid_size=40_001)
    a, b, _, weight = _brute_force_product(problem, 1.0, half=6.0, size=1_601)
    product = (a * b).reshape(-1)
    weight = weight.reshape(-1)
    keep = weight > 1e-14
    product, weight = product[keep], weight[keep] / weight[keep].sum()
    pointwise = (
        -0.5 * LOG_2PI
        - 0.5 * (problem.y[None, :] - product[:, None] * problem.x[None, :]) ** 2
    )
    lppd = torch.logsumexp(pointwise + weight.log()[:, None], 0).sum()
    first, second = weight @ pointwise, weight @ pointwise**2
    p_waic = (second - first**2).sum()
    oracle = problem.waic()
    for name, value in (
        ("lppd", lppd),
        ("p_waic", p_waic),
        ("waic", -2 * (lppd - p_waic)),
    ):
        record_analytic(
            case_id=f"oracle.product.waic.{name}",
            method=f"oracle.ProductModel.waic.{name}",
            model="y = a b x + eps, true ab = 0",
            expected=float(value),
            actual=oracle[name],
            unit="native_ic" if name == "waic" else "nats",
            oracle="mixture over the a-grid of Gaussian b | a moments vs dense 2D grid",
            atol=1e-5,
            rtol=1e-8,
            category="exact",
        )


def test_regular_learning_coefficient_is_half_dimension():
    """Linear regression with d = 2: lambda_hat -> d/2 = 1 at the WBIC temperature."""
    for n, tolerance in ((1_000, 0.01), (10_000, 0.002)):
        x, y = make_regression_data(n, [0.5, -0.3], sigma=1.0, seed=0)
        value = LinearGaussian(x, y, sigma=1.0, alpha=1.0).learning_coefficient()
        assert value == pytest.approx(1.0, abs=tolerance)


def test_product_model_with_nonzero_truth_has_lambda_one_half():
    """True ab = 1: the zero set ab = 1 is a smooth curve, lambda = 1/2, m = 1."""
    x, y = make_regression_data(10_000, [1.0], sigma=1.0, seed=0)
    problem = ProductModel(x, y, sigma=1.0, alpha=1.0, grid_size=100_001)
    assert problem.learning_coefficient() == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_singular_product_model_lambda_is_far_below_half_dimension(seed):
    """True ab = 0: lambda = 1/2 with multiplicity m = 2, versus d/2 = 1.

    At finite n the exact lambda_hat lies below 1/2 and approaches it slowly
    (about 0.33 at n = 100 and 0.41 at n = 100 000 averaged over datasets). Sampler
    tests therefore compare with this finite-n oracle, not with the limit 1/2.
    """
    values = []
    for n in (1_000, 10_000):
        x, y = make_regression_data(n, [0.0], sigma=1.0, seed=seed)
        problem = ProductModel(x, y, sigma=1.0, alpha=1.0, grid_size=100_001)
        values.append(problem.learning_coefficient())
    assert all(0.3 < value < 0.5 for value in values)
