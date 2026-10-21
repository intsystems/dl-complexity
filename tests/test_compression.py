"""Public mathematical and binary round-trip checks for compression primitives."""

import json
import math
import struct

import pytest
import torch
from torch import nn

from dl_complexity.compression import (
    AroraMatrixProject,
    BinaryModelCodec,
    EncodedModel,
    LowRankApproximation,
    MagnitudePruning,
    UniformQuantization,
    make_projection_helpers,
    matrix_project,
    nominal_projection_count,
    uniform_quantize,
)


def linear_with_weights(weights):
    tensor = torch.tensor(weights, dtype=torch.float64)
    model = nn.Linear(tensor.shape[1], tensor.shape[0], bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(tensor)
    return model


def test_magnitude_pruning_known_mask_and_no_mutation():
    model = linear_with_weights([[1, -1, 2, 3]])
    result = MagnitudePruning(0.5).apply(model)
    torch.testing.assert_close(
        result.weight, torch.tensor([[0, 0, 2, 3]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        model.weight, torch.tensor([[1, -1, 2, 3]], dtype=torch.float64)
    )
    assert result.weight.dtype == model.weight.dtype
    assert result.weight is not model.weight


def test_low_rank_diagonal_known_error():
    model = linear_with_weights([[3, 0], [0, 2]])
    result = LowRankApproximation(1).apply(model)
    torch.testing.assert_close(
        result.weight, torch.diag(torch.tensor([3.0, 0.0], dtype=torch.float64))
    )
    torch.testing.assert_close(
        torch.linalg.matrix_norm(model.weight - result.weight),
        torch.tensor(2.0, dtype=torch.float64),
    )


def test_quantization_rounding_and_constant_range():
    values = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    quantized = uniform_quantize(values, bits=1)
    assert quantized.codes.tolist() == [0, 0, 1]  # nearest-even tie
    assert quantized.dequantize().dtype == values.dtype
    torch.testing.assert_close(
        quantized.dequantize(), torch.tensor([0.0, 0.0, 1.0], dtype=values.dtype)
    )
    constant = torch.full((3,), 2.5, dtype=torch.float64)
    torch.testing.assert_close(uniform_quantize(constant, 3).dequantize(), constant)
    model = linear_with_weights([[0.0, 0.5, 1.0]])
    transformed = UniformQuantization(1).apply(model)
    torch.testing.assert_close(transformed.weight, quantized.dequantize().reshape(1, 3))


@pytest.mark.parametrize("encoding", ["dense", "sparse", "low_rank", "quantized"])
def test_codec_roundtrip_and_actual_inclusive_message_length(encoding):
    model = linear_with_weights([[0, 1], [2, 3]])
    model.eval()
    model.weight.requires_grad_(False)
    codec = BinaryModelCodec(encoding, rank=1, bits=2)
    encoded = codec.encode(model)
    assert encoded.length_bits == len(encoded.payload) * 8
    assert encoded.length_bits > model.weight.numel() * model.weight.element_size() * 8
    header_length = struct.unpack(">Q", encoded.payload[5:13])[0]
    header = json.loads(encoded.payload[13 : 13 + header_length])
    charged_data = sum(sum(record["part_lengths"]) for record in header["state"])
    assert len(encoded.payload) == 13 + header_length + charged_data
    # A differently configured decoder can reconstruct from the message alone.
    rng_before = torch.random.get_rng_state().clone()
    restored = BinaryModelCodec().decode(encoded)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert not restored.training and not restored.weight.requires_grad
    assert restored.weight.dtype == torch.float64
    if encoding == "low_rank":
        expected = LowRankApproximation(1).apply(model).weight
    elif encoding == "quantized":
        expected = uniform_quantize(model.weight, 2).dequantize()
    else:
        expected = model.weight
    torch.testing.assert_close(restored.weight, expected)
    torch.testing.assert_close(
        model.weight, torch.tensor([[0, 1], [2, 3]], dtype=torch.float64)
    )


def test_quantized_byte_and_sparse_mask_payload_counts():
    model = linear_with_weights([[0, 1], [2, 3]])
    message = BinaryModelCodec("quantized", bits=2).encode(model).payload
    length = struct.unpack(">Q", message[5:13])[0]
    assert message[13 + length :] == bytes([0b00011011])
    sparse = (
        BinaryModelCodec("sparse").encode(linear_with_weights([[0, 2], [0, 3]])).payload
    )
    length = struct.unpack(">Q", sparse[5:13])[0]
    assert sparse[13 + length] == 0b01010000
    assert len(sparse) - (13 + length) == 1 + 2 * 8  # mask byte + two float64s


def test_codec_architecture_scalar_buffers_and_dtype():
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=torch.float64),
        nn.BatchNorm1d(2, dtype=torch.float64),
        nn.ReLU(),
    )
    model.eval()
    model[0].train()
    with torch.no_grad():
        model[1].num_batches_tracked.fill_(7)
    restored = BinaryModelCodec().decode(BinaryModelCodec().encode(model))
    assert not restored.training and restored[0].training and not restored[1].training
    assert restored[1].num_batches_tracked.item() == 7
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_codec_rejects_ambiguous_architecture_alias_and_corruption():
    class Custom(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError, match="unsupported architecture"):
        BinaryModelCodec().encode(Custom())
    layer = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="tied parameters"):
        BinaryModelCodec().encode(nn.Sequential(layer, layer))
    encoded = BinaryModelCodec().encode(linear_with_weights([[1]]))
    for bad in (encoded.payload[:-1], encoded.payload + b"x", b"bad"):
        with pytest.raises(ValueError):
            BinaryModelCodec().decode(EncodedModel(bad))
    with pytest.raises(ValueError, match="disagrees"):
        BinaryModelCodec().decode(
            encoded, model_factory=lambda: nn.Linear(2, 2, bias=False)
        )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_dense_codec_preserves_floating_dtype_without_numpy(dtype):
    model = nn.Linear(2, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -1.0]], dtype=dtype))
    restored = BinaryModelCodec().decode(BinaryModelCodec().encode(model))
    assert restored.weight.dtype == dtype
    torch.testing.assert_close(restored.weight, model.weight, rtol=0, atol=0)


def test_quantization_invalid_range_is_not_silently_overflowed():
    with pytest.raises(ValueError, match="overflows"):
        uniform_quantize(torch.tensor([-1e308, 1e308], dtype=torch.float64), 2)


def test_arora_matrix_project_exact_basis_and_fixed_alphabet():
    matrix = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    # Two orthogonal Rademacher rows give exact identity under /K.
    helper = torch.tensor([[[1.0, 1.0]], [[1.0, -1.0]]], dtype=torch.float64)
    alphabet = torch.arange(-3, 4, dtype=torch.float64)
    projection = matrix_project(matrix, helper, alphabet)
    torch.testing.assert_close(projection.reconstructed, matrix, rtol=0, atol=0)
    assert projection.coefficient_indices.tolist() == [6, 2]  # coefficients +3, -1
    assert projection.parameter_count == 2 and projection.values_per_parameter == 7
    assert projection.max_coefficient_rounding_error.item() == 0
    assert nominal_projection_count(0.5, 0.1) == math.ceil(math.log(10) / 0.25)


def test_arora_project_model_and_local_rng():
    model = nn.Sequential(
        linear_with_weights([[1.0, 0.0], [0.0, 1.0]]),
        nn.ReLU(),
        linear_with_weights([[1.0, -1.0]]),
    )
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    helpers = make_projection_helpers(model, 4, seed=123)
    assert torch.equal(torch.random.get_rng_state(), rng)
    again = make_projection_helpers(model, 4, seed=123)
    assert all(torch.equal(helpers[name], again[name]) for name in helpers)
    projection = AroraMatrixProject(
        helpers, torch.arange(-8, 9, dtype=torch.float64)
    ).project(model)
    assert projection.parameter_count == 8 and projection.values_per_parameter == 17
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in state_before.items()
    )
    assert projection.model is not model


@pytest.mark.parametrize(
    "mutation",
    ["hook", "pre_hook", "forward", "call", "buffer", "nonpersistent_buffer", "tied"],
)
def test_arora_project_rejects_predictors_outside_finite_decoder(mutation):
    model = nn.Sequential(
        linear_with_weights([[1, 0], [0, 1]]),
        nn.ReLU(),
        linear_with_weights([[1, 0], [0, 1]]),
    )
    helpers = make_projection_helpers(model, 4, seed=42)
    if mutation == "hook":
        model[0].register_forward_hook(lambda module, args, output: output + 1)
    elif mutation == "pre_hook":
        model[0].register_forward_pre_hook(lambda module, args: (args[0] + 1,))
    elif mutation == "forward":
        model.forward = lambda x: x
    elif mutation == "call":
        model._call_impl = lambda x: x
    elif mutation in ("buffer", "nonpersistent_buffer"):
        model.register_buffer(
            "unaccounted_state", torch.eye(2), persistent=mutation == "buffer"
        )
    elif mutation == "tied":
        model[2].weight = model[0].weight
    with pytest.raises(ValueError):
        AroraMatrixProject(helpers, torch.arange(-4, 5, dtype=torch.float64)).project(
            model
        )


def test_arora_decode_uses_clean_bound_schema_and_restores_rng():
    model = linear_with_weights([[1, 2]])
    helper = torch.tensor([[[1.0, 1.0]], [[1.0, -1.0]]], dtype=torch.float64)
    projected = AroraMatrixProject(
        {"weight": helper}, torch.arange(-3, 4, dtype=torch.float64)
    ).project(model)
    projected.model.eval()  # Harmless mode change is not a structural mutation.
    rng = torch.random.get_rng_state().clone()
    decoded = projected.decode()
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert decoded is not projected.model and type(decoded) is nn.Linear
    torch.testing.assert_close(decoded.weight, model.weight, rtol=0, atol=0)
