"""Sampler on small networks and API edge cases (S-20, S-21)."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dl_complexity import DataSource, ModelContext
from dl_complexity.bayesian import WBIC
from dl_complexity.criteria import WAIC
from dl_complexity.errors import InputError
from dl_complexity.integrations.torch import (
    CategoricalLikelihood,
    GaussianLikelihood,
    Langevin,
    prepare_fit_statistics,
    prepare_posterior_log_likelihood,
    prepare_tempered_likelihood,
    sample_posterior,
)

DT = torch.float64


def blobs(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-1.5, 0.0], [1.5, 0.0], [0.0, 1.5]], dtype=DT)
    y = torch.randint(3, (n,), generator=g)
    x = centers[y] + torch.randn(n, 2, generator=g, dtype=DT)
    return x, y


def trained_mlp(x, y, width=8, steps=300):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, width, dtype=DT), nn.Tanh(), nn.Linear(width, 3, dtype=DT)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()
    return model


@pytest.fixture(scope="module")
def mlp_problem():
    x, y = blobs()
    model = trained_mlp(x, y)
    context = ModelContext(model, likelihood=CategoricalLikelihood(), state_id="mlp")
    return context, DataSource.from_tensors(x, y, data_id="blobs-200"), x, y


def test_mala_on_mlp_gives_finite_waic_and_wbic(mlp_problem):
    context, data, x, y = mlp_problem
    k = sum(p.numel() for p in context.model.parameters())
    posterior = prepare_posterior_log_likelihood(
        context,
        data=data,
        sampler=Langevin(1e-3, 150, burn_in=150, thin=2, metropolis=True, seed=0),
        prior_precision=1.0,
    )
    assert posterior.log_likelihood.shape == (150, 200)
    assert bool(torch.isfinite(posterior.log_likelihood).all())
    waic = WAIC().estimate(inputs=posterior)
    p_waic = float(waic.quantities["p_waic"].value)
    assert math.isfinite(float(waic.primary.value))
    assert 0 < p_waic < k  # far fewer effective than nominal parameters
    tempered = prepare_tempered_likelihood(
        context,
        data=data,
        sampler=Langevin(5e-3, 150, burn_in=150, thin=2, metropolis=True, seed=1),
        prior_precision=1.0,
    )
    wbic = float(WBIC().estimate(inputs=tempered).primary.value)
    with torch.no_grad():
        map_nll = -float(prepare_fit_statistics(context, data=data).log_likelihood)
    # MAP loss instead of nL_n(w0), so only a rough bound
    lambda_hat = (wbic - map_nll) / math.log(200)
    assert 0 < lambda_hat < k / 2


def test_mala_acceptance_falls_as_the_step_grows(mlp_problem):
    context, data, _, _ = mlp_problem
    rates = [
        sample_posterior(
            context,
            data=data,
            sampler=Langevin(step, 60, burn_in=20, metropolis=True, seed=2),
            prior_precision=1.0,
        ).acceptance_rate
        for step in (1e-4, 1e-3, 3e-2)
    ]
    assert rates[0] > 0.9 and rates[0] >= rates[1] > rates[2]


def test_sgld_on_mlp_with_minibatches_is_finite(mlp_problem):
    context, _, x, y = mlp_problem
    batches = DataSource(
        [(x[i : i + 50], y[i : i + 50]) for i in range(0, 200, 50)], n_observations=200
    )
    draws = sample_posterior(
        context,
        data=batches,
        sampler=Langevin(1e-4, 20, burn_in=20, thin=3, minibatch=True, seed=3),
        prior_precision=1.0,
    )
    assert draws.acceptance_rate is None
    assert bool(torch.isfinite(draws.parameters).all())
    assert bool(torch.isfinite(draws.log_likelihood).all())
    assert draws.log_likelihood.shape == (20, 200)


def test_dropout_and_batchnorm_are_sampled_in_eval_mode():
    x, y = blobs(n=60, seed=4)
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(2, 6, dtype=DT),
        nn.BatchNorm1d(6, dtype=DT),
        nn.ReLU(),
        nn.Dropout(0.9),
        nn.Linear(6, 3, dtype=DT),
    )
    with torch.no_grad():
        model[1].running_mean.copy_(torch.linspace(-1, 1, 6, dtype=DT))
    model.train()
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    context = ModelContext(model, likelihood=CategoricalLikelihood())
    draws = sample_posterior(
        context,
        data=DataSource([(x[:25], y[:25]), (x[25:], y[25:])]),
        sampler=Langevin(1e-3, 4, thin=2, seed=5),
        prior_precision=1.0,
    )
    assert all(m.training for m in model.modules())  # caller's mode untouched
    shapes = {name: p.shape for name, p in model.named_parameters()}
    reference = copy.deepcopy(model).eval()
    for vector, row in zip(draws.parameters, draws.log_likelihood):
        state, offset = {}, 0
        for name in draws.names:
            size = math.prod(shapes[name])
            state[name] = vector[offset : offset + size].reshape(shapes[name])
            offset += size
        with torch.no_grad():
            logits = torch.func.functional_call(reference, state, (x,))
        expected = F.log_softmax(logits, -1).gather(-1, y[:, None]).squeeze(-1)
        torch.testing.assert_close(row, expected)
    for name, value in model.named_buffers():
        assert torch.equal(value, buffers[name]), name


def test_tied_parameters_are_sampled_once():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(1, 1, bias=False, dtype=DT)
            self.second = nn.Linear(1, 1, bias=False, dtype=DT)
            self.second.weight = self.first.weight

        def forward(self, x):
            return self.second(self.first(x))

    x = torch.linspace(-1, 1, 10, dtype=DT)[:, None]
    draws = sample_posterior(
        ModelContext(Tied(), likelihood=GaussianLikelihood()),
        data=DataSource.from_tensors(x, 0.5 * x),
        sampler=Langevin(1e-3, 3, seed=6),
        prior_precision=1.0,
    )
    assert draws.names == ("first.weight",)
    assert draws.parameters.shape == (3, 1)


def test_float32_model_is_sampled_in_float32():
    x, y = blobs(n=40, seed=7)
    model = nn.Linear(2, 3)
    draws = sample_posterior(
        ModelContext(model, likelihood=CategoricalLikelihood()),
        data=DataSource.from_tensors(x.float(), y),
        sampler=Langevin(1e-3, 3, metropolis=True, seed=7),
        prior_precision=1.0,
    )
    assert draws.parameters.dtype == draws.log_likelihood.dtype == torch.float32


def test_unseeded_chain_reports_a_seed_that_reproduces_it():
    x, y = blobs(n=40, seed=8)
    context = ModelContext(
        nn.Linear(2, 3, dtype=DT), likelihood=CategoricalLikelihood()
    )
    data = DataSource.from_tensors(x, y)
    first = sample_posterior(
        context, data=data, sampler=Langevin(1e-3, 3), prior_precision=1.0
    )
    second = sample_posterior(
        context, data=data, sampler=Langevin(1e-3, 3), prior_precision=1.0
    )
    assert isinstance(first.seed, int) and first.seed != second.seed
    again = sample_posterior(
        context,
        data=data,
        sampler=Langevin(1e-3, 3, seed=first.seed),
        prior_precision=1.0,
    )
    assert torch.equal(again.parameters, first.parameters)


@pytest.mark.parametrize("minibatch", [False, True], ids=["ula", "sgld"])
def test_diverging_chain_is_stopped_with_a_clear_error(minibatch):
    # quadratic potential with curvature 40: h = 10 blows up quickly
    x = torch.ones(40, 1, dtype=DT)
    context = ModelContext(nn.Linear(1, 1, dtype=DT), likelihood=GaussianLikelihood())
    with pytest.raises(InputError, match="diverged"):
        sample_posterior(
            context,
            data=DataSource([(x[:20], x[:20]), (x[20:], x[20:])]),
            sampler=Langevin(10.0, 300, minibatch=minibatch, seed=10),
            prior_precision=1.0,
        )


def test_models_that_cannot_be_sampled_are_rejected():
    x, y = blobs(n=20, seed=11)
    data = DataSource.from_tensors(x, y)
    sampler = Langevin(1e-3, 2, seed=0)
    frozen = nn.Linear(2, 3, dtype=DT).requires_grad_(False)
    mixed = nn.Sequential(nn.Linear(2, 3, dtype=DT), nn.Linear(3, 3))
    for model in (frozen, mixed):
        with pytest.raises(InputError):
            sample_posterior(
                ModelContext(model, likelihood=CategoricalLikelihood()),
                data=data,
                sampler=sampler,
                prior_precision=1.0,
            )


def test_reduced_loss_and_wrong_sampler_type_are_rejected():
    x, y = blobs(n=20, seed=12)
    data = DataSource.from_tensors(x, y)
    model = nn.Linear(2, 3, dtype=DT)

    def mean_log_prob(prediction, target):
        return CategoricalLikelihood().log_prob(prediction, target).mean()

    with pytest.raises(InputError, match="pointwise"):
        sample_posterior(
            ModelContext(model, likelihood=mean_log_prob),
            data=data,
            sampler=Langevin(1e-3, 2, seed=0),
            prior_precision=1.0,
        )
    context = ModelContext(model, likelihood=CategoricalLikelihood())
    for function in (
        sample_posterior,
        prepare_posterior_log_likelihood,
        prepare_tempered_likelihood,
    ):
        with pytest.raises(InputError):
            function(
                context, data=data, sampler={"step_size": 1e-3}, prior_precision=1.0
            )
    one_shot = DataSource(iter([(x, y)]))
    with pytest.raises(InputError, match="reusable"):
        prepare_tempered_likelihood(
            context, data=one_shot, sampler=Langevin(1e-3, 2), prior_precision=1.0
        )
    with pytest.raises(InputError, match="n > 1"):
        prepare_tempered_likelihood(
            context,
            data=DataSource.from_tensors(x[:1], y[:1]),
            sampler=Langevin(1e-3, 2),
            prior_precision=1.0,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"metropolis": 1},
        {"minibatch": "yes"},
        {"seed": -1},
        {"seed": True},
        {"seed": 1.5},
        {"burn_in": 1.0},
    ],
    ids=str,
)
def test_langevin_rejects_non_boolean_flags_and_bad_seeds(settings):
    with pytest.raises(InputError):
        Langevin(1e-3, 2, **settings)


def test_total_steps_counts_burn_in_and_thinning():
    assert Langevin(1e-3, 7, burn_in=5, thin=3).total_steps == 5 + 7 * 3


def test_global_rng_is_restored_even_if_the_data_source_shuffles():
    x, y = blobs(n=40, seed=13)
    context = ModelContext(
        nn.Linear(2, 3, dtype=DT), likelihood=CategoricalLikelihood()
    )

    def shuffled():  # like DataLoader(shuffle=True): consumes the global RNG
        order = torch.randperm(40)
        return [(x[order[:20]], y[order[:20]]), (x[order[20:]], y[order[20:]])]

    state = torch.random.get_rng_state().clone()
    sample_posterior(
        context,
        data=DataSource(shuffled, n_observations=40),
        sampler=Langevin(1e-3, 3, minibatch=True, seed=0),
        prior_precision=1.0,
    )
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "mode", [{"metropolis": True}, {"minibatch": True}], ids=["mala", "sgld"]
)
def test_burn_in_and_thinning_select_states_of_the_same_chain(mode):
    """Draws are exactly the steps burn_in + j * thin of one chain."""
    x, y = blobs(n=40, seed=14)
    context = ModelContext(
        nn.Linear(2, 3, dtype=DT), likelihood=CategoricalLikelihood()
    )
    data = DataSource([(x[:20], y[:20]), (x[20:], y[20:])], n_observations=40)
    every = sample_posterior(
        context,
        data=data,
        sampler=Langevin(1e-3, 11, seed=15, **mode),
        prior_precision=1.0,
    )
    thinned = sample_posterior(
        context,
        data=data,
        sampler=Langevin(1e-3, 3, burn_in=2, thin=3, seed=15, **mode),
        prior_precision=1.0,
    )
    steps = [2 + 3, 2 + 6, 2 + 9]  # 1-based step numbers of the recorded states
    rows = [s - 1 for s in steps]
    assert torch.equal(thinned.parameters, every.parameters[rows])
    assert torch.equal(thinned.log_likelihood, every.log_likelihood[rows])


def test_observation_count_must_match_and_stay_constant():
    x, y = blobs(n=40, seed=16)
    context = ModelContext(
        nn.Linear(2, 3, dtype=DT), likelihood=CategoricalLikelihood()
    )
    with pytest.raises(InputError, match="Observed n"):
        sample_posterior(
            context,
            data=DataSource([(x, y)], n_observations=50),
            sampler=Langevin(1e-3, 2, seed=0),
            prior_precision=1.0,
        )
    calls = []

    def shrinking():  # the second pass silently drops observations
        calls.append(1)
        size = 40 if len(calls) == 1 else 30
        return [(x[:size], y[:size])]

    with pytest.raises(InputError, match="number of observations"):
        sample_posterior(
            context,
            data=DataSource(shrinking),
            sampler=Langevin(1e-3, 2, seed=0),
            prior_precision=1.0,
        )


def test_malformed_sources_and_uncopyable_models_are_rejected():
    import threading

    x, y = blobs(n=20, seed=17)
    sampler = Langevin(1e-3, 2, seed=0)
    context = ModelContext(
        nn.Linear(2, 3, dtype=DT), likelihood=CategoricalLikelihood()
    )
    for function in (sample_posterior, prepare_tempered_likelihood):
        with pytest.raises(InputError, match="DataSource"):
            function(context, data=[(x, y)], sampler=sampler, prior_precision=1.0)
        with pytest.raises(InputError, match="inputs, targets"):
            function(
                context,
                data=DataSource([(x, y, None)]),
                sampler=sampler,
                prior_precision=1.0,
            )
    with pytest.raises(InputError, match="no observations"):
        sample_posterior(
            context, data=DataSource([]), sampler=sampler, prior_precision=1.0
        )

    class WithLock(nn.Linear):
        def __init__(self):
            super().__init__(2, 3, dtype=DT)
            self.lock = threading.Lock()  # not deep-copyable

    with pytest.raises(InputError, match="isolated"):
        sample_posterior(
            ModelContext(WithLock(), likelihood=CategoricalLikelihood()),
            data=DataSource.from_tensors(x, y),
            sampler=sampler,
            prior_precision=1.0,
        )


def test_numpy_scalars_are_accepted_as_real_numbers():
    np = pytest.importorskip("numpy")
    x, y = blobs(n=20, seed=18)
    draws = sample_posterior(
        ModelContext(nn.Linear(2, 3, dtype=DT), likelihood=CategoricalLikelihood()),
        data=DataSource.from_tensors(x, y),
        sampler=Langevin(np.float32(1e-3), 2, seed=0),
        prior_precision=np.float64(1.0),
        beta=np.float32(0.5),
    )
    assert draws.beta == 0.5 and isinstance(draws.beta, float)
