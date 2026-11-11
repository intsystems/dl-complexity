"""Public contract checks independent of particular complexity formulas."""

import math

import pytest
import torch
from torch import nn

from dl_complexity import (
    ComplexityEstimator,
    ComplexityResult,
    ComplexitySuite,
    DataSource,
    ModelContext,
    Provenance,
    Quantity,
    RequirementStatus,
    TrainingRecipe,
)
from dl_complexity.diagnostics import WeightInputs, WeightStatistics
from dl_complexity.errors import InputError
from dl_complexity.integrations.torch import run_training


class IdentityEstimator(ComplexityEstimator):
    input_type = WeightInputs

    def _estimate(self, inputs):
        return ComplexityResult(
            "IdentityEstimator",
            "test_only",
            {
                "value": Quantity(next(iter(inputs.weights.values())), "nats"),
            },
            "value",
        )


def test_quantity_units_normalization_and_gradient():
    x = torch.tensor(math.log(2), dtype=torch.float64, requires_grad=True)
    bits = Quantity(x, "nats").to("bits")
    torch.testing.assert_close(bits.value, x.new_tensor(1))
    bits.value.backward()
    torch.testing.assert_close(x.grad, x.new_tensor(1 / math.log(2)))
    assert Quantity(x, "nats").per_observation(2).normalization == "per_observation"
    with pytest.raises(InputError):
        Quantity(x, "native_ic").to("bits")
    with pytest.raises(InputError):
        Quantity(x, "probability").per_observation(2)
    with pytest.raises(InputError):
        Quantity(x, "nats").per_observation(2).per_observation(2)


def test_metadata_preflight_and_provenance_conflict():
    method = IdentityEstimator()
    source = WeightInputs({"a": torch.tensor(2.0)}, Provenance(state_id="old"))
    assert method.check(inputs=source).status == RequirementStatus.READY
    assert method.check().status == RequirementStatus.MISSING_INPUTS
    with pytest.raises(InputError, match="state_id"):
        method.check(ModelContext(nn.Linear(1, 1), state_id="new"), inputs=source)
    with pytest.raises(InputError, match="expects"):
        method.estimate(inputs=object())


def test_suite_collects_each_method_without_fabricating_missing_values():
    source = WeightInputs({"a": torch.tensor(2.0)})
    suite = ComplexitySuite(
        {"ready": IdentityEstimator(), "missing": IdentityEstimator()}, errors="collect"
    )
    report = suite.estimate(inputs={"ready": source})
    assert set(report) == {"ready", "missing"}
    assert report["ready"].result.primary.value == 2
    assert report["missing"].result is None
    assert report["missing"].requirements.status == RequirementStatus.MISSING_INPUTS
    with pytest.raises(InputError, match="Unknown"):
        suite.check(inputs={"other": source})
    with pytest.raises(InputError, match="preflight"):
        ComplexitySuite(suite.methods).estimate(inputs={"ready": source})


def test_shared_parameter_structure_and_weight_oracle():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.tensor([[3.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
            )
            self.alias = self.weight

    model = Shared()
    ctx = ModelContext(model)
    assert [p.alias_of for p in ctx.structure.parameters] == [None, "weight"]
    result = WeightStatistics(spectral=True).estimate(ctx)
    assert result.primary is None
    for name, expected in {
        "parameter_count": 4,
        "l1": 7,
        "l2": 5,
        "sparsity": 0.5,
        "weight.spectral": 5,
    }.items():
        torch.testing.assert_close(
            result.quantities[name].value, torch.tensor(expected, dtype=torch.float64)
        )
    result.quantities["l2"].value.backward()
    torch.testing.assert_close(model.weight.grad, model.weight.detach() / 5)


def test_recipe_is_not_a_run_and_training_is_explicit_and_isolated():
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    state = model.weight.detach().clone()
    calls = []

    def train(module, *, steps):
        calls.append(steps)
        with torch.no_grad():
            module.weight.fill_(torch.rand(()).item())
        return "finished"

    ctx = ModelContext(
        model, training=TrainingRecipe(train, "declared objective", {"steps": 2})
    )
    WeightStatistics().estimate(ctx)
    assert not calls and not ctx.runs
    rng = torch.random.get_rng_state().clone()
    run = run_training(ctx, seed=42)
    assert calls == [2] and run.returned == "finished" and run.status == "completed"
    assert run.rng_seed == 42 and run.config == {"steps": 2}
    assert len(ctx.runs) == 1 and ctx.runs[0].status == "completed"
    assert not hasattr(ctx.runs[0], "model")
    torch.testing.assert_close(model.weight, state)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_recipe_seed_and_outer_rng_seed_are_not_conflated():
    seen = []

    def train(module, *, seed):
        seen.append(seed)

    ctx = ModelContext(
        nn.Linear(1, 1), training=TrainingRecipe(train, "objective", {"seed": 7})
    )
    run = run_training(ctx, seed=42)
    assert seen == [7] and run.config["seed"] == 7 and run.rng_seed == 42


def test_failed_recipe_records_failure_and_preserves_exception_rng():
    def train(module):
        torch.rand(())
        raise RuntimeError("expected training failure")

    ctx = ModelContext(nn.Linear(1, 1), training=TrainingRecipe(train, "objective"))
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="expected training failure"):
        run_training(ctx, seed=0)
    assert ctx.runs[0].status == "failed"
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_source_preflight_never_iterates_and_invalid_result_primary():
    class Spy:
        def __iter__(self):
            raise AssertionError("preflight consumed data")

    data = DataSource(Spy(), n_observations=1)
    report = IdentityEstimator().check(data=data)
    assert not report.ready
    assert not data.one_shot
    with pytest.raises(InputError):
        ComplexityResult("method", "kind", {"x": Quantity(1.0, "nats")}, "missing")
