"""Tests for the WAIC/WBIC sampler (written before the sampler itself).

Target: p(w) ~ exp(-beta nL_n(w) - alpha/2 |w|^2 - gamma/2 |w - w0|^2).
Exact answers come from sampling_oracles.py. Tolerances were tuned on many
seeds and still catch the usual bugs (no beta, no prior, sqrt(h) noise, ...).
"""

import importlib
import math

import pytest
import torch
from torch import nn

from dl_complexity import DataSource, ModelContext
from dl_complexity.bayesian import WBIC, TemperedLikelihood
from dl_complexity.criteria import WAIC, PosteriorLogLikelihood
from dl_complexity.errors import InputError
from dl_complexity.integrations.torch import GaussianLikelihood
from sampling_oracles import LinearGaussian, ProductModel, make_regression_data

SEED = 2026

# strong prior on purpose: losing the prior or beta is then easy to see
LINEAR_N, LINEAR_SIGMA, LINEAR_ALPHA = 200, 0.7, 50.0
LINEAR_WEIGHTS = [0.8, -0.5, 0.0]
# weak prior for the lambda check
WEAK_ALPHA = 1.0
PRODUCT_N, PRODUCT_SIGMA, PRODUCT_ALPHA = 200, 1.0, 1.0


@pytest.fixture(scope="module")
def sampling():
    """Import inside the fixture: without the module only these tests fail."""
    return importlib.import_module("dl_complexity.integrations.torch.sampling")


# ----------------------------------------------------------------------------
# Reference problems
# ----------------------------------------------------------------------------


def linear_problem(n=LINEAR_N, d=3, seed=11):
    x, y = make_regression_data(n, LINEAR_WEIGHTS[:d], sigma=LINEAR_SIGMA, seed=seed)
    return LinearGaussian(x, y, sigma=LINEAR_SIGMA, alpha=LINEAR_ALPHA), x, y


def linear_context(problem, weights=None, *, bias=False):
    with torch.random.fork_rng():
        model = nn.Linear(problem.d, 1, bias=bias, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_((problem.mle() if weights is None else weights)[None])
        if bias:
            model.bias.fill_(0.1)
    return ModelContext(
        model,
        likelihood=GaussianLikelihood(LINEAR_SIGMA),
        state_id="linear-mle",
        likelihood_id=f"normal-sigma{LINEAR_SIGMA}",
    )


def mala_step(problem, beta, localization=0.0):
    """Proposal standard deviation sqrt(2h) equal to the smallest posterior sd."""
    _, covariance = problem.posterior(beta, localization)
    return 0.5 * float(torch.linalg.eigvalsh(covariance).min())


class Product(nn.Module):
    """y = a * b * x: two scalar parameters, one identifiable product."""

    def __init__(self, a, b):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a), dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float64))

    def forward(self, x):
        return self.a * self.b * x


def product_problem():
    x, y = make_regression_data(PRODUCT_N, [0.0], sigma=PRODUCT_SIGMA, seed=7)
    oracle = ProductModel(
        x, y, sigma=PRODUCT_SIGMA, alpha=PRODUCT_ALPHA, grid_size=40_001
    )
    return oracle, x, y


def product_context(problem):
    c = float(problem.x @ problem.y) / float(problem.x @ problem.x)
    start = math.sqrt(abs(c))
    return ModelContext(
        Product(math.copysign(start, c), start),
        likelihood=GaussianLikelihood(PRODUCT_SIGMA),
        state_id="product-mle",
    )


def recompute_pointwise(context, parameters, names, x, y):
    """Pointwise log-likelihood [S, n] at returned draws, via functional_call."""
    model = context.model
    shapes = {name: p.shape for name, p in model.named_parameters()}
    frozen = {n: p for n, p in model.named_parameters() if n not in names}
    rows = []
    for vector in parameters:
        state, offset = dict(frozen), 0
        for name in names:
            size = math.prod(shapes[name])
            state[name] = vector[offset : offset + size].reshape(shapes[name])
            offset += size
        prediction = torch.func.functional_call(model, state, (x,))
        rows.append(context.likelihood.log_prob(prediction, y))
    return torch.stack(rows)


# ----------------------------------------------------------------------------
# Contract: configuration and validation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_size": 0.0},
        {"step_size": -1e-3},
        {"step_size": math.nan},
        {"step_size": math.inf},
        {"num_draws": 0},
        {"num_draws": 2.0},
        {"burn_in": -1},
        {"thin": 0},
        {"num_draws": True},
    ],
)
def test_langevin_rejects_invalid_settings(sampling, kwargs):
    settings = {"step_size": 1e-3, "num_draws": 10, **kwargs}
    with pytest.raises(InputError):
        sampling.Langevin(**settings)


def test_metropolis_correction_requires_full_batch(sampling):
    with pytest.raises(InputError):
        sampling.Langevin(1e-3, 10, metropolis=True, minibatch=True)


@pytest.mark.parametrize(
    "target",
    [
        {"prior_precision": 0.0},
        {"prior_precision": -1.0},
        {"prior_precision": math.inf},
        {"beta": 0.0},
        {"beta": -0.5},
        {"localization": -1.0},
    ],
)
def test_sample_posterior_rejects_invalid_target(sampling, target):
    problem, x, y = linear_problem(n=20, d=2)
    arguments = {"prior_precision": 1.0, **target}
    with pytest.raises(InputError):
        sampling.sample_posterior(
            linear_context(problem),
            data=DataSource.from_tensors(x, y),
            sampler=sampling.Langevin(1e-3, 2, seed=0),
            **arguments,
        )


def test_sample_posterior_requires_likelihood_and_reusable_data(sampling):
    problem, x, y = linear_problem(n=20, d=2)
    sampler = sampling.Langevin(1e-3, 2, seed=0)
    no_likelihood = ModelContext(linear_context(problem).model)
    with pytest.raises(InputError):
        sampling.sample_posterior(
            no_likelihood,
            data=DataSource.from_tensors(x, y),
            sampler=sampler,
            prior_precision=1.0,
        )
    one_shot = DataSource(iter([(x, y)]), n_observations=20)
    with pytest.raises(InputError):
        sampling.sample_posterior(
            linear_context(problem), data=one_shot, sampler=sampler, prior_precision=1.0
        )


def test_waic_preparation_requires_two_draws(sampling):
    problem, x, y = linear_problem(n=20, d=2)
    with pytest.raises(InputError):
        sampling.prepare_posterior_log_likelihood(
            linear_context(problem),
            data=DataSource.from_tensors(x, y),
            sampler=sampling.Langevin(1e-3, 1, seed=0),
            prior_precision=1.0,
        )


# ----------------------------------------------------------------------------
# Contract: outputs, determinism and isolation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("metropolis", [False, True], ids=["ula", "mala"])
def test_draws_record_shapes_metadata_and_provenance(sampling, metropolis):
    problem, x, y = linear_problem(n=40, d=2)
    context = linear_context(problem, bias=True)
    data = DataSource.from_tensors(x, y, data_id="train-40")
    draws = sampling.sample_posterior(
        context,
        data=data,
        sampler=sampling.Langevin(
            1e-3, 7, burn_in=5, thin=3, metropolis=metropolis, seed=0
        ),
        prior_precision=2.0,
        beta=0.5,
        localization=0.25,
    )
    assert isinstance(draws, sampling.PosteriorDraws)
    assert draws.names == ("weight", "bias")
    assert draws.parameters.shape == (7, 3)
    assert draws.log_likelihood.shape == (7, 40)
    assert draws.parameters.dtype == draws.log_likelihood.dtype == torch.float64
    assert draws.n == 40
    assert draws.beta == 0.5
    assert draws.prior_precision == 2.0
    assert draws.localization == 0.25
    if metropolis:
        assert 0.0 < draws.acceptance_rate <= 1.0
    else:
        assert draws.acceptance_rate is None
    assert draws.provenance.state_id == "linear-mle"
    assert draws.provenance.data_id == "train-40"
    assert draws.provenance.likelihood_id == context.likelihood_id


@pytest.mark.parametrize("minibatch", [False, True], ids=["full", "minibatch"])
def test_log_likelihood_rows_belong_to_returned_parameters(sampling, minibatch):
    """Each row is the full-data pointwise log-likelihood at that draw."""
    problem, x, y = linear_problem(n=40, d=2)
    context = linear_context(problem, bias=True)
    batches = [(x[i : i + 10], y[i : i + 10]) for i in range(0, 40, 10)]
    draws = sampling.sample_posterior(
        context,
        data=DataSource(batches, n_observations=40),
        sampler=sampling.Langevin(1e-3, 6, thin=2, minibatch=minibatch, seed=1),
        prior_precision=1.0,
    )
    expected = recompute_pointwise(context, draws.parameters, draws.names, x, y)
    torch.testing.assert_close(draws.log_likelihood, expected)
    assert len(torch.unique(draws.parameters, dim=0)) == 6


def test_frozen_parameters_are_excluded_and_kept_fixed(sampling):
    problem, x, y = linear_problem(n=40, d=2)
    context = linear_context(problem, bias=True)
    context.model.bias.requires_grad_(False)
    draws = sampling.sample_posterior(
        context,
        data=DataSource.from_tensors(x, y),
        sampler=sampling.Langevin(1e-3, 4, seed=2),
        prior_precision=1.0,
    )
    assert draws.names == ("weight",)
    assert draws.parameters.shape == (4, 2)
    expected = recompute_pointwise(context, draws.parameters, draws.names, x, y)
    torch.testing.assert_close(draws.log_likelihood, expected)


def test_same_seed_reproduces_and_different_seed_differs(sampling):
    problem, x, y = linear_problem(n=40, d=2)
    data = DataSource.from_tensors(x, y)

    def run(seed):
        return sampling.sample_posterior(
            linear_context(problem),
            data=data,
            sampler=sampling.Langevin(1e-3, 5, burn_in=3, metropolis=True, seed=seed),
            prior_precision=1.0,
        )

    first, again, other = run(10), run(10), run(11)
    assert torch.equal(first.parameters, again.parameters)
    assert torch.equal(first.log_likelihood, again.log_likelihood)
    assert not torch.equal(first.parameters, other.parameters)


def test_sampling_leaves_model_mode_weights_and_global_rng_untouched(sampling):
    problem, x, y = linear_problem(n=40, d=2)
    context = linear_context(problem, bias=True)
    context.model.train()
    before = {k: v.clone() for k, v in context.model.state_dict().items()}
    flags = [p.requires_grad for p in context.model.parameters()]
    rng = torch.random.get_rng_state().clone()
    sampling.sample_posterior(
        context,
        data=DataSource.from_tensors(x, y),
        sampler=sampling.Langevin(1e-3, 5, metropolis=True, seed=3),
        prior_precision=1.0,
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    assert context.model.training
    assert [p.requires_grad for p in context.model.parameters()] == flags
    assert all(p.grad is None for p in context.model.parameters())
    for name, value in context.model.state_dict().items():
        assert torch.equal(value, before[name])


def test_full_batch_gradient_sums_over_all_batches(sampling):
    """Splitting the data into batches must not change the full-batch chain."""
    problem, x, y = linear_problem(n=40, d=2)
    sampler = sampling.Langevin(2e-3, 20, burn_in=10, seed=4)
    whole = sampling.sample_posterior(
        linear_context(problem),
        data=DataSource.from_tensors(x, y),
        sampler=sampler,
        prior_precision=3.0,
        beta=0.7,
    )
    split = sampling.sample_posterior(
        linear_context(problem),
        data=DataSource(
            lambda: [(x[i : i + 8], y[i : i + 8]) for i in range(0, 40, 8)]
        ),
        sampler=sampler,
        prior_precision=3.0,
        beta=0.7,
    )
    assert split.n == 40
    torch.testing.assert_close(split.parameters, whole.parameters)
    torch.testing.assert_close(split.log_likelihood, whole.log_likelihood)


def test_waic_preparation_is_the_beta_one_draw_matrix(sampling):
    problem, x, y = linear_problem(n=40, d=2)
    data = DataSource.from_tensors(x, y, data_id="train-40")
    sampler = sampling.Langevin(1e-3, 8, metropolis=True, seed=5)
    prepared = sampling.prepare_posterior_log_likelihood(
        linear_context(problem), data=data, sampler=sampler, prior_precision=1.0
    )
    draws = sampling.sample_posterior(
        linear_context(problem), data=data, sampler=sampler, prior_precision=1.0
    )
    assert isinstance(prepared, PosteriorLogLikelihood)
    assert draws.beta == 1.0
    torch.testing.assert_close(prepared.log_likelihood, draws.log_likelihood)
    assert prepared.provenance.data_id == "train-40"
    assert WAIC().check(inputs=prepared).ready
    WAIC().estimate(inputs=prepared)


def test_wbic_preparation_uses_inverse_log_n_temperature(sampling):
    problem, x, y = linear_problem(n=40, d=2)
    # No declared n_observations: the preparation must count observations itself.
    data = DataSource(lambda: [(x[:25], y[:25]), (x[25:], y[25:])])
    sampler = sampling.Langevin(1e-3, 8, metropolis=True, seed=6)
    prepared = sampling.prepare_tempered_likelihood(
        linear_context(problem), data=data, sampler=sampler, prior_precision=1.0
    )
    beta = 1 / math.log(40)
    draws = sampling.sample_posterior(
        linear_context(problem),
        data=data,
        sampler=sampler,
        prior_precision=1.0,
        beta=beta,
    )
    assert isinstance(prepared, TemperedLikelihood)
    assert prepared.n == 40
    assert prepared.beta == pytest.approx(beta, rel=1e-12)
    assert prepared.log_likelihood.shape == (8,)
    torch.testing.assert_close(prepared.log_likelihood, draws.log_likelihood.sum(1))
    WBIC().estimate(inputs=prepared)


# ----------------------------------------------------------------------------
# Numerical agreement with exact oracles
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linear_runs(sampling):
    """MALA on the strong-prior linear problem at beta = 1 and beta = 1/log n."""
    problem, x, y = linear_problem()
    data = DataSource.from_tensors(x, y)
    runs = {}
    for label, beta in (("beta1", 1.0), ("wbic", 1 / math.log(LINEAR_N))):
        runs[label] = sampling.sample_posterior(
            linear_context(problem),
            data=data,
            sampler=sampling.Langevin(
                mala_step(problem, beta),
                1_500,
                burn_in=300,
                thin=2,
                metropolis=True,
                seed=SEED,
            ),
            prior_precision=LINEAR_ALPHA,
            beta=beta,
        )
    return problem, runs


@pytest.fixture(scope="module")
def regular_wbic_run(sampling):
    """Weak-prior linear problem at beta = 1/log n, where lambda_hat ~ d/2."""
    x, y = make_regression_data(LINEAR_N, LINEAR_WEIGHTS, sigma=LINEAR_SIGMA, seed=11)
    problem = LinearGaussian(x, y, sigma=LINEAR_SIGMA, alpha=WEAK_ALPHA)
    beta = 1 / math.log(LINEAR_N)
    with torch.random.fork_rng():
        model = nn.Linear(problem.d, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(problem.mle()[None])
    draws = sampling.prepare_tempered_likelihood(
        ModelContext(model, likelihood=GaussianLikelihood(LINEAR_SIGMA)),
        data=DataSource.from_tensors(x, y),
        sampler=sampling.Langevin(
            mala_step(problem, beta),
            1_500,
            burn_in=300,
            thin=2,
            metropolis=True,
            seed=SEED + 4,
        ),
        prior_precision=WEAK_ALPHA,
    )
    return problem, draws


def assert_gaussian_moments(record_analytic, case, draws, mean, covariance, *, z, rtol):
    """Mean within z posterior sd and variance within rtol, per coordinate."""
    sd = covariance.diag().sqrt()
    sample_mean = draws.parameters.mean(0)
    sample_var = draws.parameters.var(0)
    for i in range(mean.numel()):
        record_analytic(
            case_id=f"{case}.mean[{i}]",
            method="sample_posterior",
            model="Bayesian linear regression, known noise",
            expected=mean[i],
            actual=sample_mean[i],
            unit="parameter",
            oracle=f"closed-form posterior mean; tolerance {z} posterior sd",
            atol=z * float(sd[i]),
            rtol=0.0,
            category="approximation",
        )
        record_analytic(
            case_id=f"{case}.var[{i}]",
            method="sample_posterior",
            model="Bayesian linear regression, known noise",
            expected=covariance[i, i],
            actual=sample_var[i],
            unit="parameter^2",
            oracle="closed-form posterior variance",
            atol=0.0,
            rtol=rtol,
            category="approximation",
        )


@pytest.mark.parametrize("label", ["beta1", "wbic"])
def test_mala_matches_linear_gaussian_posterior(label, linear_runs, record_analytic):
    problem, runs = linear_runs
    draws = runs[label]
    mean, covariance = problem.posterior(draws.beta)
    assert 0.2 < draws.acceptance_rate <= 1.0
    assert_gaussian_moments(
        record_analytic,
        f"mala.linear.{label}",
        draws,
        mean,
        covariance,
        z=0.25,
        rtol=0.3,
    )


def test_mala_waic_matches_linear_gaussian_oracle(linear_runs, record_analytic):
    problem, runs = linear_runs
    result = WAIC().estimate(
        inputs=PosteriorLogLikelihood(runs["beta1"].log_likelihood)
    )
    oracle = problem.waic()
    for name, atol, unit in (
        ("p_waic", 0.3 * oracle["p_waic"], "dimensionless"),
        ("lppd", 0.25, "nats"),
        ("waic", 0.8, "native_ic"),
    ):
        record_analytic(
            case_id=f"mala.linear.waic.{name}",
            method=f"WAIC.{name} after sample_posterior",
            model="Bayesian linear regression, known noise",
            expected=oracle[name],
            actual=result.quantities[name].value,
            unit=unit,
            oracle="population WAIC of the exact Gaussian posterior",
            atol=atol,
            rtol=0.0,
            category="approximation",
        )


def test_mala_wbic_recovers_regular_learning_coefficient(
    regular_wbic_run, record_analytic
):
    """Regular model, d = 3: WBIC matches the oracle and lambda_hat ~ d/2 = 1.5."""
    problem, tempered = regular_wbic_run
    wbic = float(WBIC().estimate(inputs=tempered).primary.value)
    expected = problem.expected_total_nll(tempered.beta)
    reference = problem.total_nll(problem.mle())
    oracle_lambda = problem.learning_coefficient()
    record_analytic(
        case_id="mala.linear.wbic",
        method="WBIC after prepare_tempered_likelihood",
        model="Bayesian linear regression, known noise, weak prior",
        expected=expected,
        actual=wbic,
        unit="nats",
        oracle="nL_n(m_beta) + tr(X^T X Sigma_beta) / (2 sigma^2)",
        atol=0.15 / tempered.beta,
        rtol=0.0,
        category="approximation",
    )
    record_analytic(
        case_id="mala.linear.lambda_hat",
        method="(WBIC - nL_n(w_mle)) / log n",
        model="Bayesian linear regression, known noise, weak prior",
        expected=oracle_lambda,
        actual=(wbic - reference) * tempered.beta,
        unit="dimensionless",
        oracle="exact finite-n lambda_hat; d/2 = 1.5 asymptotically",
        atol=0.15,
        rtol=0.0,
        category="approximation",
        details={"half_dimension": problem.d / 2},
    )
    assert oracle_lambda == pytest.approx(problem.d / 2, abs=0.05)


def test_localized_target_matches_closed_form(sampling, record_analytic):
    """Localization pulls towards the context's current weights, not towards zero."""
    problem, x, y = linear_problem()
    center = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
    beta, gamma = 1 / math.log(LINEAR_N), 200.0
    draws = sampling.sample_posterior(
        linear_context(problem, center),
        data=DataSource.from_tensors(x, y),
        sampler=sampling.Langevin(
            mala_step(problem, beta, gamma),
            1_500,
            burn_in=300,
            thin=2,
            metropolis=True,
            seed=SEED + 1,
        ),
        prior_precision=LINEAR_ALPHA,
        beta=beta,
        localization=gamma,
    )
    mean, covariance = problem.posterior(beta, gamma, center)
    assert_gaussian_moments(
        record_analytic,
        "mala.linear.localized",
        draws,
        mean,
        covariance,
        z=0.25,
        rtol=0.3,
    )


def test_minibatch_sgld_scales_gradient_to_full_data(sampling, record_analytic):
    """10 batches of 20; without scaling the variance is ~5x too big.

    thin = 3 is coprime with the 10-step batch cycle (thin = 5 biases the mean).
    """
    problem, x, y = linear_problem()
    batches = [(x[i : i + 20], y[i : i + 20]) for i in range(0, LINEAR_N, 20)]
    _, covariance = problem.posterior(1.0)
    step = 0.1 * float(torch.linalg.eigvalsh(covariance).min())
    draws = sampling.sample_posterior(
        linear_context(problem),
        data=DataSource(batches, n_observations=LINEAR_N),
        sampler=sampling.Langevin(
            step, 2_000, burn_in=500, thin=3, minibatch=True, seed=SEED + 2
        ),
        prior_precision=LINEAR_ALPHA,
    )
    assert draws.log_likelihood.shape == (2_000, LINEAR_N)
    mean, covariance = problem.posterior(1.0)
    assert_gaussian_moments(
        record_analytic,
        "sgld.linear.minibatch",
        draws,
        mean,
        covariance,
        z=0.3,
        rtol=0.4,  # unadjusted Langevin keeps an O(h) variance bias
    )


def test_sgld_weights_every_observation_once_with_unequal_batches(
    sampling, record_analytic
):
    """Batches of 150 and 50 pulling w in opposite directions.

    True mean ~ +0.34; with n / B_b scaling the chain goes to ~ -0.18 (11 sd).
    """
    x_a, y_a = make_regression_data(150, [1.0], sigma=LINEAR_SIGMA, seed=21)
    x_b, y_b = make_regression_data(50, [-1.0], sigma=LINEAR_SIGMA, seed=22)
    x, y = torch.cat([x_a, x_b]), torch.cat([y_a, y_b])
    problem = LinearGaussian(x, y, sigma=LINEAR_SIGMA, alpha=WEAK_ALPHA)
    mean, covariance = problem.posterior(1.0)
    step = 0.1 * float(torch.linalg.eigvalsh(covariance).min())
    with torch.random.fork_rng():
        model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(problem.mle()[None])
    draws = sampling.sample_posterior(
        ModelContext(model, likelihood=GaussianLikelihood(LINEAR_SIGMA)),
        data=DataSource([(x_a, y_a), (x_b, y_b)], n_observations=200),
        sampler=sampling.Langevin(
            step, 2_000, burn_in=500, thin=3, minibatch=True, seed=SEED + 5
        ),
        prior_precision=WEAK_ALPHA,
    )
    sd = float(covariance[0, 0].sqrt())
    # Stationary mean of the wrongly weighted chain: likelihood of batch b
    # multiplied by (n / B_b) / (number of batches) = 2/3 and 2.
    weights = torch.cat([torch.full((150,), 2 / 3), torch.full((50,), 2.0)]).double()
    xw = x[:, 0] * weights
    wrong_mean = float((xw @ y) / (xw @ x[:, 0] + WEAK_ALPHA * LINEAR_SIGMA**2))
    assert abs(float(mean[0]) - wrong_mean) > 5 * sd  # the test can tell them apart
    # Only the mean is checked: with batches this different the stochastic
    # gradient noise inflates the SGLD variance well beyond the O(h) bias.
    record_analytic(
        case_id="sgld.linear.unequal_batches.mean",
        method="sample_posterior",
        model="Bayesian linear regression, batches of 150 and 50",
        expected=mean[0],
        actual=draws.parameters[:, 0].mean(),
        unit="parameter",
        oracle="closed-form posterior mean; tolerance 1 posterior sd",
        atol=1.0 * sd,
        rtol=0.0,
        category="approximation",
        details={"n_over_batch_weighting_mean": wrong_mean},
    )


@pytest.fixture(scope="module")
def product_runs(sampling):
    """MALA on y = abx (true ab = 0) at beta = 1 and beta = 1/log n."""
    problem, x, y = product_problem()
    data = DataSource.from_tensors(x, y)
    runs = {}
    for label, beta, step in (
        ("beta1", 1.0, 3e-3),
        ("wbic", 1 / math.log(PRODUCT_N), 1e-2),
    ):
        runs[label] = sampling.sample_posterior(
            product_context(problem),
            data=data,
            sampler=sampling.Langevin(
                step, 2_000, burn_in=500, thin=2, metropolis=True, seed=SEED + 3
            ),
            prior_precision=PRODUCT_ALPHA,
            beta=beta,
        )
    return problem, runs


def test_mala_wbic_detects_singular_product_model(product_runs, record_analytic):
    """lambda_hat follows the exact finite-n value (~0.32), far below d/2 = 1."""
    problem, runs = product_runs
    draws = runs["wbic"]
    wbic = float(-draws.log_likelihood.sum(1).mean())
    oracle_lambda = problem.learning_coefficient()
    lambda_hat = (wbic - problem.min_total_nll()) * draws.beta
    record_analytic(
        case_id="mala.product.lambda_hat",
        method="(WBIC - min nL_n) / log n",
        model="y = a b x + eps, true ab = 0 (lambda = 1/2, m = 2)",
        expected=oracle_lambda,
        actual=lambda_hat,
        unit="dimensionless",
        oracle="exact finite-n lambda_hat by 1D quadrature",
        atol=0.08,
        rtol=0.0,
        category="approximation",
        details={"regular_value": 1.0, "asymptotic_value": 0.5},
    )
    assert oracle_lambda < 0.5 and lambda_hat < 0.6


def test_mala_waic_on_singular_product_model(product_runs, record_analytic):
    """p_waic stays well below the parameter count 2 of the singular model."""
    problem, runs = product_runs
    result = WAIC().estimate(
        inputs=PosteriorLogLikelihood(runs["beta1"].log_likelihood)
    )
    oracle = problem.waic()
    record_analytic(
        case_id="mala.product.p_waic",
        method="WAIC.p_waic after sample_posterior",
        model="y = a b x + eps, true ab = 0",
        expected=oracle["p_waic"],
        actual=result.quantities["p_waic"].value,
        unit="dimensionless",
        oracle="mixture of Gaussian b | a moments over the a-grid",
        atol=0.3 * oracle["p_waic"],
        rtol=0.0,
        category="approximation",
    )
    record_analytic(
        case_id="mala.product.waic",
        method="WAIC after sample_posterior",
        model="y = a b x + eps, true ab = 0",
        expected=oracle["waic"],
        actual=result.primary.value,
        unit="native_ic",
        oracle="mixture of Gaussian b | a moments over the a-grid",
        atol=0.6,
        rtol=0.0,
        category="approximation",
    )
    assert oracle["p_waic"] < 1.5
