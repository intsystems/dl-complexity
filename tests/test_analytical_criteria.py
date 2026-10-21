"""Analytical model-level checks, independent of the library's numeric reducers.

The known-variance Gaussian checks count only fitted mean coefficients.  A
separate plug-in-variance case explicitly counts the additional fitted scale.
Bernoulli uses two Linear logits but one identifiable statistical parameter.
These are small correctness examples, not model-selection benchmarks.
"""

import math

import pytest
import torch
from torch import nn

from dl_complexity import DataSource, ModelContext
from dl_complexity.criteria import AIC, BIC, HQIC, WAIC, PosteriorLogLikelihood
from dl_complexity.integrations.torch import (
    CategoricalLikelihood,
    GaussianLikelihood,
    prepare_fit_statistics,
)


def _record_fitted_model(
    record_analytic,
    *,
    case_id,
    model_name,
    context,
    data,
    expected_ll,
    k,
    n,
    dtype,
    oracle,
    details,
    parameter_count="trainable",
):
    """Compare both public routes with a scalar Python likelihood oracle."""
    atol, rtol = (5e-12, 3e-14) if dtype == torch.float64 else (2e-5, 3e-7)
    before = {name: value.clone() for name, value in context.model.state_dict().items()}
    flags = [module.training for module in context.model.modules()]
    stats = prepare_fit_statistics(context, data=data, parameter_count=parameter_count)
    assert stats.k == k and stats.n == n
    assert stats.log_likelihood.dtype == dtype
    record_analytic(
        case_id=f"{case_id}.log_likelihood",
        method="prepare_fit_statistics",
        model=model_name,
        expected=expected_ll,
        actual=stats.log_likelihood,
        unit="nats",
        oracle=oracle,
        atol=atol,
        rtol=rtol,
        category="exact",
        details={**details, "k": k, "n": n, "dtype": str(dtype)},
    )
    penalties = (
        (AIC, 2 * k),
        (BIC, k * math.log(n)),
        (HQIC, 2 * k * math.log(math.log(n))),
    )
    for cls, penalty in penalties:
        estimator = cls(parameter_count=parameter_count)
        expected = -2 * expected_ll + penalty
        supplied = estimator.estimate(inputs=stats)
        direct = estimator.estimate(context, data=data)
        for route, result in (("supplied", supplied), ("context", direct)):
            assert result.kind == "information_criterion"
            assert result.primary.unit == "native_ic"
            assert result.primary.normalization == "total"
            assert result.config["k"] == k and result.config["n"] == n
            record_analytic(
                case_id=f"{case_id}.{cls.__name__}.{route}",
                method=cls.__name__,
                model=model_name,
                expected=expected,
                actual=result.primary.value,
                unit="native_ic",
                oracle=f"{oracle}; IC=-2*ell+penalty, penalty={penalty!r}",
                atol=2 * atol,
                rtol=rtol,
                category="exact",
                details={
                    **details,
                    "k": k,
                    "n": n,
                    "route": route,
                    "dtype": str(dtype),
                },
            )
            assert result.quantities["deviance"].value.item() == pytest.approx(
                -2 * expected_ll, abs=2 * atol, rel=rtol
            )
            assert result.quantities["penalty"].value.item() == pytest.approx(
                penalty, abs=atol, rel=rtol
            )
            per_observation = result.primary.per_observation(n)
            assert per_observation.normalization == "per_observation"
            assert per_observation.value.item() == pytest.approx(
                expected / n, abs=2 * atol / n, rel=rtol
            )
    assert [module.training for module in context.model.modules()] == flags
    for name, value in context.model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize(
    "observations,sigma,dtype",
    [
        ([1.0, 2.0, 4.0, 5.0], 0.5, torch.float64),
        ([1.0, 2.0, 4.0, 5.0], 1.0, torch.float64),
        ([-1.0, 0.0, 2.0, 3.0, 5.0], 2.0, torch.float64),
        ([0.0, 0.5, 1.0, 1.5], 0.25, torch.float32),
        ([1.0, 2.0, 4.0, 5.0] * 4, 2.0, torch.float32),
    ],
)
def test_gaussian_mean_mle_known_variance(observations, sigma, dtype, record_analytic):
    """Evaluate the fitted Normal mean, including the density normalization."""
    n = len(observations)
    mean = math.fsum(observations) / n
    rss = math.fsum((value - mean) ** 2 for value in observations)
    expected_ll = -n * (math.log(sigma) + 0.5 * math.log(2 * math.pi))
    expected_ll -= rss / (2 * sigma**2)
    model = nn.Linear(1, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(mean)
    x = torch.ones(n, 1, dtype=dtype)
    y = torch.tensor(observations, dtype=dtype).reshape(n, 1)
    # Unequal reusable batches ensure n means observations, not batch count.
    data = DataSource(((x[:1], y[:1]), (x[1:], y[1:])), n_observations=n)
    context = ModelContext(model, likelihood=GaussianLikelihood(sigma=sigma))
    _record_fitted_model(
        record_analytic,
        case_id=f"normal_mean.n{n}.sigma{sigma}.{dtype}",
        model_name="Normal mean MLE, known variance",
        context=context,
        data=data,
        expected_ll=expected_ll,
        k=1,
        n=n,
        dtype=dtype,
        oracle="mu_hat=sum(y)/n; ell=-n*log(sigma*sqrt(2*pi))-RSS/(2*sigma^2)",
        details={"mu_hat": mean, "sigma_known": sigma, "rss": rss},
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gaussian_mean_and_variance_mle_explicit_dimension(dtype, record_analytic):
    """Count a fitted scale even though it is passed via a likelihood adapter."""
    observations = [0.0, 1.0, 3.0, 4.0, 7.0]
    n = len(observations)
    mean = math.fsum(observations) / n
    rss = math.fsum((value - mean) ** 2 for value in observations)
    variance_mle = rss / n  # MLE denominator n, not the unbiased denominator n-1.
    sigma_mle = math.sqrt(variance_mle)
    expected_ll = -0.5 * n * (math.log(2 * math.pi * variance_mle) + 1)
    model = nn.Linear(1, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(mean)
    context = ModelContext(model, likelihood=GaussianLikelihood(sigma=sigma_mle))
    data = DataSource.from_tensors(
        torch.ones(n, 1, dtype=dtype),
        torch.tensor(observations, dtype=dtype).reshape(n, 1),
    )
    _record_fitted_model(
        record_analytic,
        case_id=f"normal_mean_scale.{dtype}",
        model_name="Normal mean and variance MLE, explicit k=2",
        context=context,
        data=data,
        expected_ll=expected_ll,
        k=2,
        n=n,
        dtype=dtype,
        oracle="mu_hat=mean(y); sigma_hat^2=RSS/n; ell=-n/2*[log(2*pi*sigma_hat^2)+1]",
        details={"mu_hat": mean, "variance_mle": variance_mle, "rss": rss},
        parameter_count=2,
    )
    # The adapter cannot infer that a supplied sigma was fitted on this sample.
    assert prepare_fit_statistics(context, data=data).k == 1


@pytest.mark.parametrize(
    "repeats,sigma,dtype",
    [
        (1, 0.5, torch.float64),
        (3, 2.0, torch.float64),
        (1, 1.5, torch.float32),
        (3, 0.75, torch.float32),
    ],
)
def test_scalar_ols_mle_with_intercept(repeats, sigma, dtype, record_analytic):
    """Obtain OLS from scalar centered sums, not a library least-squares call."""
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0] * repeats
    ys = [0.0, -1.0, 5.0, 3.0, 8.0] * repeats
    n = len(xs)
    x_mean, y_mean = math.fsum(xs) / n, math.fsum(ys) / n
    numerator = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.fsum((x - x_mean) ** 2 for x in xs)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    residuals = [y - intercept - slope * x for x, y in zip(xs, ys)]
    assert math.fsum(residuals) == pytest.approx(0.0, abs=1e-14)
    assert math.fsum(x * r for x, r in zip(xs, residuals)) == pytest.approx(
        0.0, abs=1e-14
    )
    rss = math.fsum(r**2 for r in residuals)
    expected_ll = -n * (math.log(sigma) + 0.5 * math.log(2 * math.pi))
    expected_ll -= rss / (2 * sigma**2)
    model = nn.Linear(1, 1, dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(slope)
        model.bias.fill_(intercept)
    x = torch.tensor(xs, dtype=dtype).reshape(n, 1)
    y = torch.tensor(ys, dtype=dtype).reshape(n, 1)
    data = DataSource(
        lambda: (
            (x[start : start + 2], y[start : start + 2]) for start in range(0, n, 2)
        ),
        n_observations=n,
    )
    _record_fitted_model(
        record_analytic,
        case_id=f"ols.n{n}.sigma{sigma}.{dtype}",
        model_name="Scalar Gaussian OLS with intercept, known variance",
        context=ModelContext(model, likelihood=GaussianLikelihood(sigma=sigma)),
        data=data,
        expected_ll=expected_ll,
        k=2,
        n=n,
        dtype=dtype,
        oracle="b_hat=sum[(x-xbar)(y-ybar)]/sum[(x-xbar)^2]; a_hat=ybar-b_hat*xbar; Normal ell",
        details={
            "intercept": intercept,
            "slope": slope,
            "sigma_known": sigma,
            "rss": rss,
        },
    )


@pytest.mark.parametrize(
    "successes,n,dtype",
    [
        (2, 6, torch.float64),
        (6, 8, torch.float64),
        (6, 10, torch.float32),
        (3, 10, torch.float32),
    ],
)
def test_bernoulli_mle_identifiable_dimension(successes, n, dtype, record_analytic):
    """Softmax has two stored logits but only one identifiable Bernoulli p."""
    probability = successes / n
    expected_ll = successes * math.log(probability)
    expected_ll += (n - successes) * math.log1p(-probability)
    model = nn.Linear(1, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [[0.0], [math.log(probability / (1 - probability))]], dtype=dtype
            )
        )
    x = torch.ones(n, 1, dtype=dtype)
    y = torch.tensor([1] * successes + [0] * (n - successes), dtype=torch.long)
    context = ModelContext(model, likelihood=CategoricalLikelihood())
    data = DataSource(((x[:3], y[:3]), (x[3:], y[3:])), n_observations=n)
    _record_fitted_model(
        record_analytic,
        case_id=f"bernoulli.s{successes}.n{n}.{dtype}",
        model_name="Bernoulli MLE represented by two softmax logits",
        context=context,
        data=data,
        expected_ll=expected_ll,
        k=1,
        n=n,
        dtype=dtype,
        oracle="p_hat=s/n; ell=s*log(p_hat)+(n-s)*log(1-p_hat); identifiable k=1",
        details={
            "successes": successes,
            "p_hat": probability,
            "raw_parameter_count": 2,
        },
        parameter_count=1,
    )
    assert prepare_fit_statistics(context, data=data).k == 2


def test_bernoulli_mle_checkpoint_keeps_analytical_score(tmp_path, record_analytic):
    """A weights-only checkpoint plus architecture reproduces the same MLE score."""
    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.0], [math.log(3)]], dtype=torch.float64))
    checkpoint = tmp_path / "bernoulli_mle.pt"
    torch.save(model.state_dict(), checkpoint)
    context = ModelContext.from_checkpoint(
        checkpoint,
        model_factory=lambda: nn.Linear(1, 2, bias=False, dtype=torch.float64),
        likelihood=CategoricalLikelihood(),
    )
    n, successes = 8, 6
    data = DataSource.from_tensors(
        torch.ones(n, 1, dtype=torch.float64),
        torch.tensor([1] * successes + [0] * (n - successes)),
    )
    _record_fitted_model(
        record_analytic,
        case_id="bernoulli.checkpoint",
        model_name="Bernoulli MLE loaded from state_dict",
        context=context,
        data=data,
        expected_ll=6 * math.log(0.75) + 2 * math.log(0.25),
        k=1,
        n=n,
        dtype=torch.float64,
        oracle="p_hat=6/8; ell=6*log(3/4)+2*log(1/4); statistical k=1",
        details={"successes": successes, "p_hat": 0.75, "checkpoint": "state_dict"},
        parameter_count=1,
    )


@pytest.mark.parametrize("repetitions", [1, 2, 16])
def test_exact_two_hypothesis_bernoulli_waic_finite_draw_correction(
    repetitions, record_analytic
):
    """Check the implemented finite-S reducer against an exact posterior model.

    A prior giving p=1/4 and p=3/4 equal mass is specified before seeing balanced
    observations [0,1,0,1].  Their likelihoods match, so posterior masses remain
    1/2. Enumerating each state R times reproduces exact posterior expectations,
    but the public sample-variance convention still multiplies variance by S/(S-1).
    These deterministic enumerations are not a claim of independent MCMC draws.
    """
    ys = [0, 1, 0, 1]
    hypotheses = [0.25, 0.75]
    likelihoods = [math.prod(p if y else 1 - p for y in ys) for p in hypotheses]
    posterior_mass = [
        0.5 * likelihood / math.fsum(0.5 * l for l in likelihoods)
        for likelihood in likelihoods
    ]
    assert posterior_mass == pytest.approx([0.5, 0.5], abs=1e-15)
    sample_count, n = 2 * repetitions, len(ys)
    matrix = torch.tensor(
        [[math.log(p if y else 1 - p) for y in ys] for p in hypotheses] * repetitions,
        dtype=torch.float64,
    )
    expected_lppd = n * math.log(0.5)
    population_variance = n * math.log(3) ** 2 / 4
    correction = sample_count / (sample_count - 1)
    finite_p_waic = correction * population_variance
    expected_waic = -2 * (expected_lppd - finite_p_waic)
    result = WAIC().estimate(inputs=PosteriorLogLikelihood(matrix))
    details = {
        "hypotheses": hypotheses,
        "prior_mass": [0.5, 0.5],
        "posterior_mass": posterior_mass,
        "S": sample_count,
        "N": n,
        "population_p_waic": population_variance,
        "sample_variance_multiplier": correction,
        "enumerated_not_random_draws": True,
    }
    for name, expected, unit in (
        ("lppd", expected_lppd, "nats"),
        ("p_waic", finite_p_waic, "dimensionless"),
        ("waic", expected_waic, "native_ic"),
    ):
        record_analytic(
            case_id=f"two_hypothesis_waic.S{sample_count}.{name}",
            method=f"WAIC.{name}",
            model="Exact two-hypothesis Bernoulli posterior",
            expected=expected,
            actual=result.quantities[name].value,
            unit=unit,
            oracle="lppd=N*ln(1/2); Var_post(log p)=ln(3)^2/4 per datum; finite variance *=S/(S-1)",
            atol=5e-12,
            rtol=3e-14,
            category="finite_draw",
            details=details,
        )
    # The ddof=1 reducer is intentionally not equal to population-posterior WAIC.
    population_waic = -2 * (expected_lppd - population_variance)
    assert result.primary.value.item() - population_waic == pytest.approx(
        2 * population_variance / (sample_count - 1), abs=5e-12
    )
