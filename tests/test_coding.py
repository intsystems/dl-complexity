"""Small mathematical oracles for public code-length and causal protocols."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dl_complexity.coding import (
    BlockwisePrequential,
    OnlinePrequential,
    PrequentialEntry,
    PrequentialLedger,
    ReplayPrequential,
    TemperatureScaler,
    TwoPartInputs,
    TwoPartMDL,
    prepare_blockwise,
    prepare_online,
    prepare_replay,
    prepare_two_part,
)
from dl_complexity.errors import InputError
from dl_complexity.results import RequirementStatus


def test_two_part_exact_units_components_and_gradient():
    nll = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    inputs = TwoPartInputs(model_bits=10, data_nll=nll, n=2)
    result = TwoPartMDL().estimate(inputs=inputs)
    torch.testing.assert_close(result.primary.value, nll + 10 * math.log(2))
    assert result.primary.unit == "nats"
    assert result.kind == "two_part_code"
    assert result.quantities["per_observation"].normalization == "per_observation"
    result.primary.value.backward()
    torch.testing.assert_close(nll.grad, torch.tensor(1.0, dtype=torch.float64))
    bits = TwoPartMDL(unit="bits").estimate(inputs=inputs)
    torch.testing.assert_close(bits.primary.value, nll / math.log(2) + 10)


@pytest.mark.parametrize(
    "inputs",
    [
        TwoPartInputs(-1, 2, 3),
        TwoPartInputs(1, -2, 3),
        TwoPartInputs(1, 2, 0),
        TwoPartInputs(1, 2, 3, data_code_convention="density"),
        TwoPartInputs(1, 2, 3, resolution_adjustment_nats=1),
    ],
)
def test_two_part_invalid_codes_are_not_silently_reinterpreted(inputs):
    with pytest.raises(InputError):
        TwoPartMDL().estimate(inputs=inputs)


def test_two_part_quantized_density_requires_resolution():
    result = TwoPartMDL().estimate(
        inputs=TwoPartInputs(
            8,
            -2.0,
            1,
            data_code_convention="quantized",
            resolution_adjustment_nats=3.0,
        )
    )
    assert result.quantities["data"].value.item() == 1.0
    assert result.kind == "two_part_density_code_approximation"


def test_ledger_manual_total_and_provenance_route():
    ledger = PrequentialLedger(
        math.log(2),
        1,
        (PrequentialEntry(1, 3, -math.log(0.8) - math.log(0.25), 1),),
        3,
        "blockwise",
    )
    result = BlockwisePrequential().estimate(inputs=ledger)
    assert result.primary.value.item() == pytest.approx(
        math.log(2) - math.log(0.8) - math.log(0.25)
    )
    bits = BlockwisePrequential(unit="bits").estimate(inputs=ledger)
    assert bits.primary.value.item() == pytest.approx(
        result.primary.value.item() / math.log(2)
    )
    assert BlockwisePrequential().check().status == RequirementStatus.MISSING_INPUTS


@pytest.mark.parametrize(
    "ledger",
    [
        PrequentialLedger(0.0, 0, (PrequentialEntry(0, 2, 1.0, 1),), 2, "online"),
        PrequentialLedger(0.0, 0, (PrequentialEntry(0, 1, 1.0, 0, 1),), 1, "online"),
        PrequentialLedger(0.0, 0, (PrequentialEntry(1, 2, 1.0, 0),), 2, "online"),
        PrequentialLedger(0.0, 0, (PrequentialEntry(0, 1, 1.0, 0),), 2, "online"),
        PrequentialLedger(
            0.0, 0, (PrequentialEntry(0, 1, 1.0, 0, replay_indices=(1,)),), 1, "online"
        ),
    ],
)
def test_ledger_rejects_future_labels_and_incomplete_intervals(ledger):
    with pytest.raises(InputError):
        OnlinePrequential().estimate(inputs=ledger)


def test_blockwise_training_receives_only_prefixes():
    x = torch.arange(4, dtype=torch.float64)[:, None]
    y = torch.tensor([0, 1, 1, 0])
    trained_prefixes = []

    def train(state, prefix_x, prefix_y):
        trained_prefixes.append(prefix_x[:, 0].tolist())
        assert torch.equal(prefix_y, y[: len(prefix_x)])
        state["prefix"] = len(prefix_x)

    def predict(state, block):
        assert int(block[0, 0]) >= state["prefix"]
        return torch.log(torch.tensor([[0.75, 0.25]], dtype=torch.float64)).expand(
            len(block), 2
        )

    ledger = prepare_blockwise(
        x,
        y,
        boundaries=[1, 3, 4],
        model_factory=dict,
        train=train,
        n_classes=2,
        predict_logits=predict,
    )
    assert trained_prefixes == [[0.0], [0.0, 1.0, 2.0]]
    result = BlockwisePrequential().estimate(inputs=ledger)
    expected = math.log(2) - 2 * math.log(0.25) - math.log(0.75)
    assert result.primary.value.item() == pytest.approx(expected)
    assert [e.trained_until for e in ledger.entries] == [1, 3]


def test_online_scores_before_update_with_known_probabilities():
    x = torch.arange(3, dtype=torch.float64)[:, None]
    y = torch.tensor([0, 1, 0])
    events = []

    def predict(state, batch):
        events.append(("predict", state["seen"]))
        assert state["seen"] == int(batch[0, 0])
        return torch.tensor([[math.log(0.6), math.log(0.4)]], dtype=torch.float64)

    def update(state, batch, labels):
        events.append(("update", int(batch[0, 0])))
        state["seen"] += len(batch)

    ledger = prepare_online(
        x,
        y,
        model_factory=lambda: {"seen": 0},
        update=update,
        predict_logits=predict,
        n_classes=2,
    )
    assert events == [
        ("predict", 0),
        ("update", 0),
        ("predict", 1),
        ("update", 1),
        ("predict", 2),
        ("update", 2),
    ]
    assert OnlinePrequential().estimate(
        inputs=ledger
    ).primary.value.item() == pytest.approx(-2 * math.log(0.6) - math.log(0.4))


def test_replay_does_not_code_rehearsal_again():
    x = torch.arange(4, dtype=torch.float64)[:, None]
    y = torch.tensor([0, 1, 0, 1])
    updates = []

    def predict(state, batch):
        assert state["seen"] == int(batch[0, 0])
        return torch.zeros(len(batch), 2, dtype=torch.float64)

    def update(state, batch, labels, replay_x, replay_y):
        assert bool((replay_x[:, 0] < int(batch[-1, 0]) + 1).all())
        updates.append((batch[:, 0].tolist(), replay_x[:, 0].tolist()))
        state["seen"] += len(batch)

    ledger = prepare_replay(
        x,
        y,
        model_factory=lambda: {"seen": 0},
        update=update,
        predict_logits=predict,
        n_classes=2,
        replay_size=3,
        seed=12,
    )
    assert len(updates) == 4
    assert [len(e.replay_indices) for e in ledger.entries] == [1, 2, 3, 3]
    assert ReplayPrequential().estimate(
        inputs=ledger
    ).primary.value.item() == pytest.approx(4 * math.log(2))


def test_replay_rejects_future_indices_before_update():
    updates = []
    with pytest.raises(InputError, match="revealed"):
        prepare_replay(
            torch.arange(3.0)[:, None],
            torch.tensor([0, 1, 0]),
            model_factory=dict,
            update=lambda *args: updates.append(args),
            predict_logits=lambda state, batch: torch.zeros(len(batch), 2),
            n_classes=2,
            replay_size=1,
            replay_policy=lambda stop: [stop],
        )
    assert updates == []


def test_temperature_one_and_update_do_not_rewrite_scored_cost():
    logits = torch.tensor([[3.0, -1.0], [2.0, -0.5]], dtype=torch.float64)
    labels = torch.tensor([1, 1])
    scaler = TemperatureScaler(steps=20)
    torch.testing.assert_close(scaler.log_probs(logits), F.log_softmax(logits, dim=-1))
    old = -scaler.log_probs(logits).gather(1, labels[:, None]).sum()
    old_copy = old.clone()
    scaler.update(logits, labels)
    torch.testing.assert_close(old, old_copy)
    assert scaler.temperature.item() > 1
    assert not scaler.temperature.requires_grad


def test_online_calibration_updates_only_after_scoring():
    events = []

    class SpyCalibration:
        def log_probs(self, logits):
            events.append("score")
            return F.log_softmax(logits, dim=-1)

        def update(self, logits, y):
            events.append("calibrate")

    def update(*args):
        events.append("train")

    ledger = prepare_online(
        torch.ones(2, 1),
        torch.tensor([0, 1]),
        model_factory=dict,
        update=update,
        n_classes=2,
        predict_logits=lambda s, x: torch.zeros(len(x), 2),
        calibrator=SpyCalibration(),
    )
    assert events == ["score", "calibrate", "train"] * 2
    assert [e.calibrated_until for e in ledger.entries] == [0, 1]


def test_predictor_module_flags_are_restored():
    model = nn.Linear(1, 2, dtype=torch.float64).train()
    ledger = prepare_online(
        torch.ones(2, 1, dtype=torch.float64),
        torch.tensor([0, 1]),
        model_factory=lambda: model,
        update=lambda *args: None,
        n_classes=2,
    )
    assert model.training
    assert ledger.n == 2


def test_blockwise_rejects_reusing_a_final_model():
    model = nn.Linear(1, 2, dtype=torch.float64)
    with pytest.raises(InputError, match="independent"):
        prepare_blockwise(
            torch.ones(3, 1, dtype=torch.float64),
            torch.tensor([0, 1, 0]),
            boundaries=[1, 2, 3],
            model_factory=lambda: model,
            train=lambda *args: None,
            n_classes=2,
        )


def test_preflight_rejects_incomplete_ledger_without_reading_cost_values():
    ledger = PrequentialLedger(
        float("nan"), 0, (PrequentialEntry(0, 1, 1.0, 0),), 2, "online"
    )
    with pytest.raises(InputError, match="incomplete"):
        OnlinePrequential().check(inputs=ledger)


def test_codec_preparation_uses_decoded_predictor_not_original():
    from dl_complexity.compression import BinaryModelCodec
    from dl_complexity.context import ModelContext
    from dl_complexity.data import DataSource
    from dl_complexity.integrations.torch.likelihood import CategoricalLikelihood

    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0], [-1.0]], dtype=torch.float64))
    original = model.weight.detach().clone()
    data = DataSource.from_tensors(
        torch.ones(2, 1, dtype=torch.float64), torch.tensor([0, 0])
    )
    context = ModelContext(
        model, likelihood=CategoricalLikelihood(), state_id="original"
    )
    codec = BinaryModelCodec(encoding="low_rank", rank=0)
    inputs = prepare_two_part(context, data=data, codec=codec)
    result = TwoPartMDL().estimate(inputs=inputs)
    torch.testing.assert_close(
        result.quantities["data"].value,
        torch.tensor(2 * math.log(2), dtype=torch.float64),
    )
    assert inputs.model_bits == codec.encode(model).length_bits
    assert inputs.provenance.state_id is None
    torch.testing.assert_close(model.weight, original)


def test_codec_preparation_preserves_each_decoded_parameter_dtype():
    import copy

    from dl_complexity.compression import EncodedModel
    from dl_complexity.context import ModelContext
    from dl_complexity.data import DataSource
    from dl_complexity.integrations.torch.likelihood import CategoricalLikelihood

    class MixedPrecisionPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.single_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            self.double_weight = nn.Parameter(
                torch.tensor(1.123456789012345, dtype=torch.float64)
            )

        def forward(self, x):
            return torch.stack(
                (x[:, 0] * self.double_weight, x[:, 0] * self.single_weight.double()),
                dim=1,
            )

    class SpyCodec:
        def encode(self, model):
            self.decoded = copy.deepcopy(model)
            return EncodedModel(b"mixed-precision-test")

        def decode(self, encoded, model_factory=None):
            return self.decoded

    model = MixedPrecisionPredictor()
    x = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    y = torch.tensor([0, 1])
    likelihood = CategoricalLikelihood()
    data = DataSource.from_tensors(x, y)
    context = ModelContext(model=model, likelihood=likelihood)
    codec = SpyCodec()
    inputs = prepare_two_part(context, data=data, codec=codec)
    assert codec.decoded.single_weight.dtype == torch.float32
    assert codec.decoded.double_weight.dtype == torch.float64
    expected = -likelihood.log_prob(model(x), y).sum()
    torch.testing.assert_close(inputs.data_nll, expected, atol=1e-14, rtol=1e-14)
