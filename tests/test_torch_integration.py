"""Checkpoint/model → likelihood → shared API, with independent numeric oracles."""

import math

import pytest
import torch
from torch import nn

from dl_complexity import ComplexitySuite, DataSource, ModelContext, RequirementStatus
from dl_complexity.criteria import AIC, BIC, HQIC, WAIC
from dl_complexity.diagnostics import WeightStatistics
from dl_complexity.errors import InputError
from dl_complexity.integrations.torch import (
    CategoricalLikelihood,
    GaussianLikelihood,
    prepare_fit_statistics,
)


def test_gaussian_context_likelihood_and_original_autograd():
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(2)
    x, y = torch.tensor([[1.0], [2.0]], dtype=torch.float64), torch.tensor(
        [[3.0], [5.0]], dtype=torch.float64
    )
    data = DataSource.from_tensors(x, y, data_id="train")
    ctx = ModelContext(
        model,
        likelihood=GaussianLikelihood(),
        state_id="fixed",
        likelihood_id="normal1",
    )
    stats = prepare_fit_statistics(ctx, data=data)
    assert stats.n == 2 and stats.k == 1
    expected_nll = 1 + math.log(2 * math.pi)
    torch.testing.assert_close(-stats.log_likelihood, x.new_tensor(expected_nll))
    stats.log_likelihood.backward()
    torch.testing.assert_close(model.weight.grad, x.new_tensor([[3.0]]))
    result = BIC().estimate(inputs=stats)
    torch.testing.assert_close(
        result.primary.value, x.new_tensor(2 * expected_nll + math.log(2))
    )
    assert result.provenance.state_id == "fixed"


def test_check_is_metadata_only_and_does_not_consume_rng_or_data():
    class SpyLinear(nn.Linear):
        def forward(self, x):
            raise AssertionError("check invoked forward")

    def batches():
        raise AssertionError("check invoked batch factory")

    ctx = ModelContext(SpyLinear(1, 1), likelihood=GaussianLikelihood())
    rng = torch.random.get_rng_state().clone()
    result = BIC().check(ctx, data=DataSource(batches, n_observations=3))
    assert result.status == RequirementStatus.READY
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not WAIC().check(ctx, data=DataSource(batches, n_observations=3)).ready


def test_isolation_restores_flags_weights_buffers_and_rng():
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, dtype=torch.float64)
            self.dropout = nn.Dropout(0.8)
            self.register_buffer("counter", torch.zeros((), dtype=torch.float64))

        def forward(self, x):
            self.counter.add_(1)
            torch.rand(())  # Deliberately stateful user forward.
            return self.dropout(self.linear(x))

    model = Stateful().train()
    model.dropout.eval()  # Mixed mode must survive, not merely root.training.
    ctx = ModelContext(model, likelihood=GaussianLikelihood())
    data = DataSource.from_tensors(
        torch.ones(3, 1, dtype=torch.float64), torch.zeros(3, 1, dtype=torch.float64)
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [m.training for m in model.modules()]
    rng = torch.random.get_rng_state().clone()
    BIC().estimate(ctx, data=data)
    assert [m.training for m in model.modules()] == flags
    assert torch.equal(rng, torch.random.get_rng_state())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_plain_checkpoint_and_categorical_suite(tmp_path):
    model = nn.Linear(1, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    ctx = ModelContext.from_checkpoint(
        path,
        model_factory=lambda: nn.Linear(1, 2, dtype=torch.float64),
        likelihood=CategoricalLikelihood(),
    )
    data = DataSource.from_tensors(
        torch.ones(4, 1, dtype=torch.float64), torch.tensor([0, 1, 0, 1])
    )
    shared = prepare_fit_statistics(ctx, data=data)
    suite = ComplexitySuite(
        {
            "aic": AIC(),
            "bic": BIC(),
            "hqic": HQIC(),
            "waic": WAIC(),
            "weights": WeightStatistics(),
        },
        errors="collect",
    )
    report = suite.estimate(
        ctx, data=data, inputs={k: shared for k in ["aic", "bic", "hqic"]}
    )
    torch.testing.assert_close(
        report["bic"].result.primary.value,
        torch.tensor(8 * math.log(2) + 4 * math.log(4), dtype=torch.float64),
    )
    assert report["weights"].result is not None
    assert report["waic"].result is None and not report["waic"].requirements.ready
    with pytest.raises(InputError, match="Factory dtype"):
        ModelContext.from_checkpoint(
            path, model_factory=lambda: nn.Linear(1, 2, dtype=torch.float32)
        )


def test_loss_alone_and_wrong_n_do_not_make_a_likelihood():
    model = nn.Linear(1, 1)
    data = DataSource.from_tensors(torch.ones(1, 1), torch.ones(1, 1))
    ctx = ModelContext(model, loss=nn.MSELoss())
    with pytest.raises(InputError):
        BIC().estimate(ctx, data=data)
    ctx.likelihood = GaussianLikelihood()
    with pytest.raises(InputError, match="Observed n"):
        prepare_fit_statistics(ctx, data=DataSource(data.batches, n_observations=2))


def test_suite_rejects_shared_one_shot_iterator_before_consumption():
    calls = []

    def once():
        calls.append("consumed")
        yield torch.ones(1, 1), torch.ones(1, 1)

    source = DataSource(once(), n_observations=1)
    suite = ComplexitySuite({"aic": AIC(), "bic": BIC()})
    with pytest.raises(InputError, match="reusable"):
        suite.estimate(
            ModelContext(nn.Linear(1, 1), likelihood=GaussianLikelihood()), data=source
        )
    assert not calls


def test_one_data_reader_plus_weights_can_use_single_iterator():
    data = DataSource(iter([(torch.ones(1, 1), torch.zeros(1, 1))]), n_observations=1)
    suite = ComplexitySuite({"bic": BIC(), "weights": WeightStatistics()})
    result = suite.estimate(
        ModelContext(nn.Linear(1, 1), likelihood=GaussianLikelihood()), data=data
    )
    assert all(outcome.result is not None for outcome in result.values())
