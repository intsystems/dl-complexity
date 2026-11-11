"""Two-part code on multi-layer networks (T-12, T-13, T-19).

The precision sweep uses a fixed teacher network with labels sampled from it,
so no training is needed.
"""

import copy
import math
import struct

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from dl_complexity import DataSource, ModelContext
from dl_complexity.coding import TwoPartMDL, prepare_two_part
from dl_complexity.compression import BinaryModelCodec
from dl_complexity.integrations.torch import CategoricalLikelihood


def mlp_with_buffers(dtype):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(6, 16, dtype=dtype),
        nn.BatchNorm1d(16, dtype=dtype),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 16, dtype=dtype),
        nn.GELU(),
        nn.Linear(16, 3, bias=False, dtype=dtype),
    )
    with torch.no_grad():
        model[1].running_mean.copy_(torch.linspace(-1, 1, 16, dtype=dtype))
        model[1].running_var.copy_(torch.linspace(0.5, 2, 16, dtype=dtype))
        model[1].num_batches_tracked.fill_(7)
        model[4].weight[model[4].weight.abs() < 0.1] = 0.0  # sparse structure
    model[0].bias.requires_grad_(False)
    model.train()
    model[3].eval()  # mixed per-module flags must survive the round trip
    return model


def outputs(model, x):
    reference = copy.deepcopy(model).eval()
    with torch.no_grad():
        return reference(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("encoding", ["dense", "sparse"])
def test_lossless_round_trip_of_mlp_with_batchnorm(encoding, dtype):
    model = mlp_with_buffers(dtype)
    codec = BinaryModelCodec(encoding)
    decoded = codec.decode(codec.encode(model))
    original_state, decoded_state = model.state_dict(), decoded.state_dict()
    assert original_state.keys() == decoded_state.keys()
    for name, tensor in original_state.items():
        assert decoded_state[name].dtype == tensor.dtype, name
        assert torch.equal(decoded_state[name], tensor), name
    assert [m.training for m in decoded.modules()] == [
        m.training for m in model.modules()
    ]
    assert [p.requires_grad for p in decoded.parameters()] == [
        p.requires_grad for p in model.parameters()
    ]
    x = torch.randn(20, 6, dtype=dtype)
    assert torch.equal(outputs(decoded, x), outputs(model, x))


def test_sparse_code_is_shorter_than_dense_for_a_pruned_network():
    model = mlp_with_buffers(torch.float32)
    zeros = int((model[4].weight == 0).sum())
    assert zeros > 50
    dense = BinaryModelCodec("dense").encode(model).length_bits
    sparse = BinaryModelCodec("sparse").encode(model).length_bits
    assert sparse < dense


@pytest.mark.parametrize("bits", [2, 4, 8, 12])
def test_quantized_round_trip_error_is_at_most_half_a_step(bits):
    model = mlp_with_buffers(torch.float32)
    codec = BinaryModelCodec("quantized", bits=bits)
    decoded = codec.decode(codec.encode(model)).state_dict()
    for name, tensor in model.state_dict().items():
        restored = decoded[name]
        assert restored.dtype == tensor.dtype and restored.shape == tensor.shape
        if not tensor.is_floating_point():
            assert torch.equal(restored, tensor), name  # num_batches_tracked
            continue
        values = tensor.double()
        step = float(values.max() - values.min()) / (2**bits - 1)
        cast = 1e-6 * float(values.abs().max())  # float32 storage after decode
        error = float((restored.double() - values).abs().max())
        assert error <= step / 2 + cast, (name, error, step)
        levels = torch.unique(restored).numel()
        assert levels <= 2**bits, name


def test_prepare_two_part_on_mlp_uses_decoded_network_likelihood():
    model = mlp_with_buffers(torch.float64)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(50, 6, generator=g, dtype=torch.float64)
    y = torch.randint(3, (50,), generator=g)
    codec = BinaryModelCodec("quantized", bits=3)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    with torch.no_grad():
        inputs = prepare_two_part(ctx, data=DataSource.from_tensors(x, y), codec=codec)
    encoded = codec.encode(model)
    decoded = codec.decode(encoded)
    expected_nll = F.cross_entropy(outputs(decoded, x), y, reduction="sum")
    original_nll = F.cross_entropy(outputs(model, x), y, reduction="sum")
    assert inputs.model_bits == encoded.length_bits == 8 * len(encoded.payload)
    torch.testing.assert_close(torch.as_tensor(inputs.data_nll), expected_nll)
    assert not torch.isclose(expected_nll, original_nll)  # 3 bits really changes it
    assert inputs.n == 50


# precision sweep

N_TRAIN, N_CLASSES = 20_000, 4


@pytest.fixture(scope="module")
def teacher_problem():
    g = torch.Generator().manual_seed(0)
    teacher = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, N_CLASSES),
    )
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (3.0 / math.sqrt(p.shape[-1])))
    x = torch.randn(N_TRAIN, 8, generator=g)
    with torch.no_grad():
        y = torch.multinomial(F.softmax(teacher(x), -1), 1, generator=g).squeeze(-1)
    ctx = ModelContext(teacher, likelihood=CategoricalLikelihood())
    data = DataSource.from_tensors(x, y)
    with torch.no_grad():
        original_nll = float(F.cross_entropy(teacher(x), y, reduction="sum"))
    rows = {}
    for bits in range(1, 17):
        codec = BinaryModelCodec("quantized", bits=bits)
        with torch.no_grad():
            inputs = prepare_two_part(ctx, data=data, codec=codec)
        result = TwoPartMDL(unit="bits").estimate(inputs=inputs)
        rows[bits] = {
            "model": float(result.quantities["model"].value),
            "data": float(result.quantities["data"].value),
            "total": float(result.primary.value),
            "gap_nats": float(inputs.data_nll) - original_nll,
        }
    with torch.no_grad():
        dense = TwoPartMDL(unit="bits").estimate(
            inputs=prepare_two_part(ctx, data=data, codec=BinaryModelCodec("dense"))
        )
    return {
        "rows": rows,
        "dense_total": float(dense.primary.value),
        "original_nll": original_nll,
        "n_params": sum(p.numel() for p in teacher.parameters()),
    }


def test_model_code_grows_with_precision_by_about_one_bit_per_weight(teacher_problem):
    rows, k = teacher_problem["rows"], teacher_problem["n_params"]
    model = [rows[b]["model"] for b in range(1, 17)]
    assert all(b > a for a, b in zip(model, model[1:]))
    mean_increment = (model[-1] - model[0]) / 15
    assert abs(mean_increment - k) <= 0.05 * k  # header JSON adds a few bytes


def test_decoded_likelihood_converges_to_the_original_network(teacher_problem):
    rows, nll = teacher_problem["rows"], teacher_problem["original_nll"]
    gaps = [rows[b]["gap_nats"] for b in range(1, 17)]
    # few bits: error drops with every bit
    assert all(a > b > 0 for a, b in zip(gaps[:5], gaps[1:5]))
    # many bits: same network up to rounding
    assert all(abs(g) <= 1e-3 * nll for g in gaps[11:])
    assert abs(gaps[15]) < abs(gaps[3]) / 1000


def test_total_two_part_length_has_an_interior_optimal_precision(teacher_problem):
    rows = teacher_problem["rows"]
    totals = {b: rows[b]["total"] for b in rows}
    best = min(totals, key=totals.get)
    assert 1 < best < 16
    assert totals[1] > totals[best] and totals[16] > totals[best]
    # past the optimum the model part dominates
    assert all(totals[b + 1] > totals[b] for b in range(best + 2, 16))


def test_float32_two_part_code_is_longer_than_the_uniform_label_code(teacher_problem):
    """float32 2-part is longer than the uniform code, as in Blier & Ollivier."""
    uniform_bits = N_TRAIN * math.log2(N_CLASSES)
    rows = teacher_problem["rows"]
    best_total = min(row["total"] for row in rows.values())
    assert teacher_problem["dense_total"] > uniform_bits
    assert best_total < uniform_bits  # coarse precision makes it compress


# conv nets


def small_cnn():
    torch.manual_seed(5)
    model = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            6, 8, kernel_size=(3, 1), padding=(1, 0), bias=False, padding_mode="reflect"
        ),
        nn.Tanh(),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Identity(),
        nn.Linear(32, 10),
        nn.Sigmoid(),
    )
    with torch.no_grad():
        model[1].running_mean.uniform_(-0.5, 0.5)
        model[1].running_var.uniform_(0.5, 1.5)
    return model.eval()


def test_lossless_round_trip_of_a_convolutional_network():
    model = small_cnn()
    codec = BinaryModelCodec("dense")
    decoded = codec.decode(codec.encode(model))
    for (name, a), (other, b) in zip(model.named_modules(), decoded.named_modules()):
        assert name == other and type(a) is type(b), name
    assert decoded[4].padding_mode == "reflect" and decoded[4].bias is None
    assert tuple(decoded[4].kernel_size) == (3, 1)
    assert tuple(decoded[6].output_size) == (2, 2)
    x = torch.randn(4, 3, 8, 8)
    assert torch.equal(outputs(decoded, x), outputs(model, x))


def test_two_part_code_of_a_convolutional_classifier_end_to_end():
    model = small_cnn()[:-1]  # drop Sigmoid: logits for the categorical likelihood
    g = torch.Generator().manual_seed(9)
    x = torch.randn(40, 3, 8, 8, generator=g)
    y = torch.randint(10, (40,), generator=g)
    ctx = ModelContext(model, likelihood=CategoricalLikelihood())
    data = DataSource.from_tensors(x, y)
    results = {}
    for encoding, bits in [("dense", 8), ("quantized", 4)]:
        codec = BinaryModelCodec(encoding, bits=bits)
        with torch.no_grad():
            inputs = prepare_two_part(ctx, data=data, codec=codec)
        decoded = codec.decode(codec.encode(model))
        expected = F.cross_entropy(outputs(decoded, x), y, reduction="sum")
        torch.testing.assert_close(torch.as_tensor(inputs.data_nll), expected)
        results[encoding] = TwoPartMDL(unit="bits").estimate(inputs=inputs)
    assert float(results["quantized"].primary.value) < float(
        results["dense"].primary.value
    )

    def split(encoding, bits):
        payload = BinaryModelCodec(encoding, bits=bits).encode(model).payload
        start = payload.index(b"{")  # JSON header follows an 8-byte length field
        header = struct.unpack(">Q", payload[start - 8 : start])[0]
        return 8 * header, 8 * (len(payload) - start - header)

    dense_header, dense_chunks = split("dense", 8)
    quant_header, quant_chunks = split("quantized", 4)
    state = model.state_dict().values()
    assert dense_chunks == sum(8 * t.numel() * t.element_size() for t in state)
    assert quant_chunks == sum(
        8 * math.ceil(4 * t.numel() / 8) if t.is_floating_point() else 64 * t.numel()
        for t in state
    )
    # min/scale/bits of every tensor are paid for too
    assert quant_header > dense_header


def test_codec_refuses_predictors_it_cannot_describe_faithfully():
    """The codec must refuse models it cannot describe exactly."""
    hooked = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    hooked[0].register_forward_hook(lambda module, args, out: 2 * out)
    patched = nn.Sequential(nn.Linear(2, 2))
    patched[0].forward = lambda x: x  # instance-level override changes the function
    infinite = nn.Sequential(nn.Linear(2, 2))
    with torch.no_grad():
        infinite[0].weight[0, 0] = float("inf")
    for model in (hooked, patched):
        with pytest.raises(ValueError, match="hook|forward"):
            BinaryModelCodec().encode(model)
    with pytest.raises(ValueError, match="finite"):
        BinaryModelCodec().encode(infinite)
    half = nn.Sequential(nn.Linear(2, 2, dtype=torch.float16))
    with pytest.raises(ValueError, match="low-rank"):
        BinaryModelCodec("low_rank", rank=1).encode(half)
    complex_model = nn.Sequential(nn.Linear(2, 2, dtype=torch.complex64))
    with pytest.raises(ValueError, match="unsupported"):
        BinaryModelCodec().encode(complex_model)
