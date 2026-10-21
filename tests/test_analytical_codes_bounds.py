"""Independent finite-model oracles for causal codes, compression and Arora.

These checks use public APIs and elementary probability/linear algebra. They do
not measure engineering speed, prove a generalization theorem by simulation,
or reproduce the noise-stability result of Arora et al. The finite decoder,
alphabet and confidence parameter are explicitly fixed in the Arora example.
"""

import itertools
import json
import math
import struct

import pytest
import torch
from torch import nn

from dl_complexity import DataSource, ModelContext
from dl_complexity.bounds import AroraCompressionBound, prepare_arora_projection
from dl_complexity.coding import (
    BlockwisePrequential,
    OnlinePrequential,
    ReplayPrequential,
    TemperatureScaler,
    TwoPartMDL,
    prepare_blockwise,
    prepare_online,
    prepare_replay,
    prepare_two_part,
)
from dl_complexity.compression import (
    AroraMatrixProject,
    BinaryModelCodec,
    LowRankApproximation,
    MagnitudePruning,
    UniformQuantization,
    uniform_quantize,
)
from dl_complexity.integrations.torch import CategoricalLikelihood


@pytest.mark.parametrize("labels", list(itertools.product((0, 1), repeat=4)))
def test_online_beta_bernoulli_exact_marginal(labels, record_analytic):
    """Laplace's rule of succession telescopes to a Beta function exactly."""
    x = torch.arange(4, dtype=torch.float64)[:, None]
    y = torch.tensor(labels)
    events = []

    def predict(state, batch):
        seen = state["ones"] + state["zeros"]
        assert seen == int(batch[0, 0])
        events.append(("predict", seen))
        p_one = (1 + state["ones"]) / (2 + seen)
        return torch.tensor([[math.log1p(-p_one), math.log(p_one)]], dtype=x.dtype)

    def update(state, batch, revealed):
        seen = state["ones"] + state["zeros"]
        events.append(("update", seen))
        state["ones"] += int(revealed.sum())
        state["zeros"] += len(revealed) - int(revealed.sum())

    ledger = prepare_online(
        x,
        y,
        model_factory=lambda: {"ones": 0, "zeros": 0},
        update=update,
        predict_logits=predict,
        n_classes=2,
    )
    result = OnlinePrequential().estimate(inputs=ledger)
    successes = sum(labels)
    failures = len(labels) - successes
    # No binomial coefficient: an ordered sequence, not the count statistic.
    expected = (
        math.lgamma(len(labels) + 2)
        - math.lgamma(successes + 1)
        - math.lgamma(failures + 1)
    )
    record_analytic(
        case_id="online_beta_bernoulli_" + "".join(map(str, labels)),
        method="OnlinePrequential",
        model="Bernoulli with Beta(1,1) posterior predictive",
        expected=expected,
        actual=result.primary.value,
        unit="nats",
        oracle="-log B(s+1,n-s+1); ordered-sequence marginal, no binomial factor",
        details={"labels": labels, "alpha": 1, "beta": 1},
    )
    assert events == [
        event for i in range(4) for event in (("predict", i), ("update", i))
    ]
    assert result.primary.unit == "nats"
    assert result.quantities["per_observation"].normalization == "per_observation"


@pytest.mark.parametrize("boundaries", [(1, 2, 3, 4), (1, 3, 4), (2, 4), (4,)])
def test_blockwise_exact_fixed_prefix_predictor(boundaries, record_analytic):
    """A block holds its prefix predictor fixed, unlike sequential Bayes updates."""
    labels = (1, 1, 0, 1)
    x = torch.arange(4, dtype=torch.float64)[:, None]
    y = torch.tensor(labels)
    prefixes = []

    def train(state, prefix_x, prefix_y):
        prefixes.append(len(prefix_x))
        state["p"] = (int(prefix_y.sum()) + 1) / (len(prefix_y) + 2)

    def predict(state, batch):
        p_one = state["p"]
        return torch.tensor(
            [[math.log1p(-p_one), math.log(p_one)]], dtype=x.dtype
        ).expand(len(batch), 2)

    ledger = prepare_blockwise(
        x,
        y,
        boundaries=boundaries,
        model_factory=dict,
        train=train,
        predict_logits=predict,
        n_classes=2,
    )
    expected = boundaries[0] * math.log(2)
    for start, stop in zip(boundaries, boundaries[1:]):
        p_one = (1 + sum(labels[:start])) / (2 + start)
        expected -= sum(
            math.log(p_one if label else 1 - p_one) for label in labels[start:stop]
        )
    result = BlockwisePrequential().estimate(inputs=ledger)
    record_analytic(
        case_id="blockwise_beta_bernoulli_" + "_".join(map(str, boundaries)),
        method="BlockwisePrequential",
        model="Bernoulli fixed posterior-mean predictor per prefix block",
        expected=expected,
        actual=result.primary.value,
        unit="nats",
        oracle="b0*ln(2) - sum_blocks sum_i ln p(y_i | fixed earlier prefix)",
        details={"labels": labels, "boundaries": boundaries},
    )
    assert prefixes == list(boundaries[:-1])
    assert [entry.trained_until for entry in ledger.entries] == list(boundaries[:-1])


@pytest.mark.parametrize("seed", [0, 17])
def test_replay_idempotent_posterior_has_exact_marginal(seed, record_analytic):
    """An idempotent sufficient statistic must not count rehearsed labels twice."""
    labels = (1, 0, 1, 1, 0)
    x = torch.arange(5, dtype=torch.float64)[:, None]
    y = torch.tensor(labels)
    replayed = []

    def predict(state, batch):
        assert len(state) == int(batch[0, 0])
        p_one = (1 + sum(state.values())) / (2 + len(state))
        return torch.tensor([[math.log1p(-p_one), math.log(p_one)]], dtype=x.dtype)

    def update(state, new_x, new_y, replay_x, replay_y):
        replayed.append(replay_x[:, 0].tolist())
        for batch, targets in ((new_x, new_y), (replay_x, replay_y)):
            for index, label in zip(batch[:, 0].tolist(), targets.tolist()):
                state[int(index)] = label

    ledger = prepare_replay(
        x,
        y,
        model_factory=dict,
        update=update,
        predict_logits=predict,
        n_classes=2,
        replay_size=3,
        seed=seed,
    )
    expected = math.lgamma(7) - math.lgamma(4) - math.lgamma(3)
    result = ReplayPrequential(unit="bits").estimate(inputs=ledger)
    record_analytic(
        case_id=f"replay_idempotent_beta_bernoulli_seed_{seed}",
        method="ReplayPrequential",
        model="Beta(1,1)-Bernoulli with unique-index sufficient statistics",
        expected=expected / math.log(2),
        actual=result.primary.value,
        unit="bits",
        oracle="-ln B(4,3)/ln(2); replay does not change an idempotent posterior",
        details={"labels": labels, "seed": seed, "replay_size": 3},
    )
    assert len(replayed) == len(labels)
    assert [len(indices) for indices in replayed] == [1, 2, 3, 3, 3]


def test_replay_weighted_updates_change_predictions_not_charged_count(record_analytic):
    """A known repeat-weighted learner gives a different, independently calculable code."""
    labels = (1, 1, 0, 1, 0)
    x = torch.arange(5, dtype=torch.float64)[:, None]
    y = torch.tensor(labels)

    def predict(state, batch):
        p_one = (1 + state[1]) / (2 + state[0] + state[1])
        return torch.tensor([[math.log1p(-p_one), math.log(p_one)]], dtype=x.dtype)

    def update(state, new_x, new_y, replay_x, replay_y):
        for label in (*new_y.tolist(), *replay_y.tolist()):
            state[label] += 1

    ledger = prepare_replay(
        x,
        y,
        model_factory=lambda: [0, 0],
        update=update,
        predict_logits=predict,
        n_classes=2,
        replay_size=1,
        replay_policy=lambda stop: (stop - 1,),
    )
    # Each latest label enters the learner twice, but each label is encoded once.
    expected = 0.0
    for i, label in enumerate(labels):
        successes = 2 * sum(labels[:i])
        p_one = (1 + successes) / (2 + 2 * i)
        expected -= math.log(p_one if label else 1 - p_one)
    result = ReplayPrequential().estimate(inputs=ledger)
    record_analytic(
        case_id="replay_repeat_weighted_bernoulli",
        method="ReplayPrequential",
        model="Bernoulli pseudo-count learner, newest observation rehearsed once",
        expected=expected,
        actual=result.primary.value,
        unit="nats",
        oracle="sum_i -ln p_i(y_i), p_i(1)=(1+2*s_before_i)/(2+2*i)",
        details={"labels": labels, "each_update_weight": 2, "encoded_observations": 5},
    )
    assert len(ledger.entries) == 5
    assert all(entry.replay_indices == (entry.stop - 1,) for entry in ledger.entries)


def test_temperature_one_exact_bernoulli_log_probabilities(record_analytic):
    """Identity temperature preserves the declared categorical probability model."""
    logits = torch.tensor([[0.0, math.log(3)], [math.log(4), 0.0]], dtype=torch.float64)
    labels = torch.tensor([1, 0])
    scaler = TemperatureScaler(temperature=1.0)
    log_probs = scaler.log_probs(logits)
    actual = -log_probs[torch.arange(2), labels].sum()
    record_analytic(
        case_id="temperature_one_bernoulli",
        method="TemperatureScaler.log_probs",
        model="Two categorical observations with p(correct)=3/4,4/5",
        expected=-math.log(3 / 4) - math.log(4 / 5),
        actual=actual,
        unit="nats",
        oracle="-ln(3/4)-ln(4/5), T=1 exactly; no fitting/sampler claim",
    )
    torch.testing.assert_close(
        log_probs.exp().sum(dim=1), torch.ones(2, dtype=logits.dtype)
    )


@pytest.mark.parametrize("encoding", ["dense", "sparse", "quantized"])
def test_two_part_actual_message_and_independent_decoded_bernoulli(
    encoding, record_analytic
):
    """Charge a self-describing message and probabilities of its decoded model."""
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    original_weights = torch.tensor([[0.0, 0.0], [0.4, 1.5]], dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(original_weights)
    xs = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 1.0))
    labels = (1, 1, 0, 1)
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(labels)
    codec = BinaryModelCodec(encoding=encoding, bits=2)
    encoded = codec.encode(model)
    decoded = codec.decode(encoded)
    header_length = struct.unpack(">Q", encoded.payload[5:13])[0]
    header = json.loads(encoded.payload[13 : 13 + header_length])
    expected_chunk_bytes = {"dense": 4 * 8, "sparse": 1 + 2 * 8, "quantized": 1}[
        encoding
    ]
    assert (
        sum(sum(record["part_lengths"]) for record in header["state"])
        == expected_chunk_bytes
    )
    expected_bits = 8 * (13 + header_length + expected_chunk_bytes)
    expected_weights = (0.5, 1.5) if encoding == "quantized" else (0.4, 1.5)
    torch.testing.assert_close(
        decoded.weight,
        torch.tensor([[0.0, 0.0], expected_weights], dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    # Binary logits are [0,z]; direct scalar logistic NLL, without PyTorch loss.
    expected_nll = sum(
        math.log1p(
            math.exp(
                (1 - 2 * label) * sum(a * b for a, b in zip(row, expected_weights))
            )
        )
        for row, label in zip(xs, labels)
    )
    inputs = prepare_two_part(
        ModelContext(model, likelihood=CategoricalLikelihood()),
        data=DataSource.from_tensors(x, y),
        codec=codec,
    )
    result = TwoPartMDL(unit="bits").estimate(inputs=inputs)
    details = {
        "encoding": encoding,
        "header_bytes": header_length,
        "tensor_chunk_bytes": expected_chunk_bytes,
        "decoded_class1_weights": expected_weights,
        "fixed_decoder": "BinaryModelCodec format version from payload",
    }
    record_analytic(
        case_id=f"two_part_{encoding}_model_bits",
        method="BinaryModelCodec.encode",
        model="Self-described float64 Linear(2,2,bias=False)",
        expected=expected_bits,
        actual=encoded.length_bits,
        unit="bits",
        oracle="8*(13+serialized_JSON_header_bytes+tensor_chunk_bytes); all header overhead charged",
        details=details,
    )
    record_analytic(
        case_id=f"two_part_{encoding}_decoded_data",
        method="prepare_two_part",
        model="Decoded two-class linear logistic predictor",
        expected=expected_nll,
        actual=inputs.data_nll,
        unit="nats",
        oracle="sum_i ln(1+exp((1-2*y_i)*(w1*x1+w2*x2))) using manually decoded weights",
        details=details,
    )
    record_analytic(
        case_id=f"two_part_{encoding}_total_bits",
        method="TwoPartMDL",
        model="Fixed binary model language plus categorical data code",
        expected=expected_bits + expected_nll / math.log(2),
        actual=result.primary.value,
        unit="bits",
        oracle="inclusive physical model bits + decoded-predictor NLL/ln(2), not universal model complexity",
        details=details,
    )
    assert result.primary.unit == "bits"
    assert result.quantities["per_observation"].normalization == "per_observation"
    torch.testing.assert_close(model.weight, original_weights, atol=0, rtol=0)


def test_magnitude_pruning_known_euclidean_distortion(record_analytic):
    """Removing the two smallest magnitudes has a known squared error."""
    model = nn.Linear(4, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0, 3.0, 4.0]], dtype=torch.float64))
    pruned = MagnitudePruning(0.5).apply(model)
    torch.testing.assert_close(
        pruned.weight, torch.tensor([[0.0, 0.0, 3.0, 4.0]], dtype=torch.float64)
    )
    record_analytic(
        case_id="magnitude_pruning_exact_error",
        method="MagnitudePruning.apply",
        model="Linear weight vector [1,-2,3,4], removal fraction 1/2",
        expected=1**2 + (-2) ** 2,
        actual=(model.weight - pruned.weight).square().sum(),
        unit="squared_weight",
        oracle="Squared Euclidean error equals sum of squares of the removed weights",
    )
    assert torch.count_nonzero(model.weight) == 4
    assert torch.count_nonzero(pruned.weight) == 2


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_low_rank_eckart_young_diagonal_oracle(rank, record_analytic):
    """The optimal Frobenius residual is the discarded singular-value energy."""
    model = nn.Linear(3, 3, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(
            torch.diag(torch.tensor([5.0, 3.0, 1.0], dtype=torch.float64))
        )
    approximated = LowRankApproximation(rank=rank).apply(model)
    expected = sum(value**2 for value in (5, 3, 1)[rank:])
    record_analytic(
        case_id=f"low_rank_eckart_young_rank_{rank}",
        method="LowRankApproximation.apply",
        model="Diagonal Linear matrix with singular values 5,3,1",
        expected=expected,
        actual=(model.weight - approximated.weight).square().sum(),
        unit="squared_weight",
        oracle="Eckart-Young: min_rank_r ||W-W_r||_F^2 = sum_{j>r} sigma_j^2",
        details={"rank": rank, "singular_values": [5, 3, 1]},
    )
    assert int(torch.linalg.matrix_rank(approximated.weight)) == rank


@pytest.mark.parametrize(
    "bits,expected_codes", [(1, [0, 0, 0, 0, 1, 1, 1]), (2, [0, 0, 1, 2, 2, 2, 3])]
)
def test_uniform_quantization_exact_grid_and_rounding(
    bits, expected_codes, record_analytic
):
    """Integer grid and ties-to-even determine the result without an optimizer."""
    values = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    tensor = torch.tensor([values], dtype=torch.float64)
    quantized = uniform_quantize(tensor, bits=bits)
    assert quantized.codes.tolist() == [expected_codes]
    scale = 3 / (2**bits - 1)
    expected_values = [code * scale for code in expected_codes]
    expected_error = sum(
        (original - decoded) ** 2 for original, decoded in zip(values, expected_values)
    )
    record_analytic(
        case_id=f"uniform_quantization_grid_{bits}_bits",
        method="uniform_quantize / UniformQuantization.apply",
        model="Seven known weights on [0,3], including half-grid ties",
        expected=expected_error,
        actual=(tensor - quantized.dequantize()).square().sum(),
        unit="squared_weight",
        oracle="codes=nearest-even((w-min)/scale); scale=3/(2^bits-1)",
        details={
            "bits": bits,
            "expected_codes": expected_codes,
            "expected_values": expected_values,
        },
    )
    model = nn.Linear(7, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(tensor)
    torch.testing.assert_close(
        UniformQuantization(bits=bits).apply(model).weight, quantized.dequantize()
    )
    torch.testing.assert_close(model.weight, tensor)


def test_arora_projection_rounded_linear_predictor_and_population_risk(record_analytic):
    """Check Algorithm 1, finite class and conditional Theorem 2.1 arithmetic.

    Fixed helpers/levels are specified independently before drawing iid inputs.
    The two-point population permits exact classification-risk enumeration. One
    successful example is not a proof of the theorem nor of noise stability.
    """
    helpers = torch.tensor([[[1.0], [1.0]], [[1.0], [-1.0]]], dtype=torch.float64)
    alphabet = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    q, r, gamma, delta, n = 2, 5, 1.0, 0.05, 100
    model = nn.Linear(1, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.1], [-1.1]], dtype=torch.float64))
    projected = AroraMatrixProject({"weight": helpers}, alphabet).project(model)
    torch.testing.assert_close(
        projected.model.weight, torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    )
    # Local iid Rademacher sample from a fully specified two-point population.
    generator = torch.Generator().manual_seed(2026)
    x = (2 * torch.randint(0, 2, (n, 1), generator=generator) - 1).to(torch.float64)
    y = (x[:, 0] < 0).long()
    certificate = prepare_arora_projection(
        model,
        projected,
        DataSource.from_tensors(x, y),
        margin=gamma,
        helper_is_data_independent=True,
        alphabet_is_data_independent=True,
    )
    bound = AroraCompressionBound(delta=delta).estimate(inputs=certificate)
    expected_penalty = math.sqrt((q * math.log(r) - math.log(delta)) / (2 * n))
    record_analytic(
        case_id="arora_fixed_linear_projection_distortion",
        method="AroraMatrixProject.project / prepare_arora_projection",
        model="Two-class bias-free Linear with fixed orthogonal Rademacher helper",
        expected=0.1,
        actual=certificate.max_logit_error,
        unit="dimensionless",
        oracle="<W,M>=[0,2.2] rounds to [0,2]; decoded W=[1,-1], max |x|=1",
        details={"q": q, "r": r, "margin": gamma, "n": n},
    )
    record_analytic(
        case_id="arora_fixed_linear_conditional_bound",
        method="AroraCompressionBound",
        model="Compressed predictor in a fixed family of at most 5^2 functions",
        expected=expected_penalty,
        actual=bound.primary.value,
        unit="probability",
        oracle="Lhat_gamma=0; sqrt((2*ln(5)+ln(20))/(2*100)); conditional Theorem 2.1 proof",
        details={
            "q": q,
            "r": r,
            "delta": delta,
            "sample_seed": 2026,
            "theorem_4_1": False,
        },
    )
    population = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    truth = torch.tensor([1, 0])
    with torch.no_grad():
        risk = (projected.model(population).argmax(dim=1) != truth).double().mean()
    record_analytic(
        case_id="arora_fixed_linear_exact_population_risk",
        method="Compressed predictor classification risk",
        model="Uniform population X in {-1,+1}, Y=1[X<0]",
        expected=0.0,
        actual=risk,
        unit="probability",
        oracle="Both possible population points are classified correctly; exhaustive 2-point enumeration",
        details={"population_probabilities": [0.5, 0.5]},
    )
    assert risk <= bound.primary.value
    assert certificate.parameter_count == q and certificate.values_per_parameter == r
    assert certificate.empirical_margin_loss == 0
    assert certificate.max_logit_error <= gamma / 2
    assert bound.primary.unit == "probability"
