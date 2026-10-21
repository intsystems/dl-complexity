"""A self-describing binary model language with exact physical length accounting.

Supported built-in architectures are described in the header, not supplied as
free architecture bits. There is no pickle execution and no fitted decoder
state. This deliberately simple code is not claimed to be an optimal MDL code.
"""

from collections import OrderedDict
from collections.abc import Callable
import json
import math
import struct
import sys

import torch
from torch import Tensor, nn

from .base import EncodedModel, ModelCodec, ModelTransform
from .quantization import uniform_quantize

_MAGIC = b"DLCM\x01"
_DTYPES = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
}


def _architecture(model: nn.Module) -> dict:
    r"""Describe a supported built-in predictor without executing its forward.

    Args:
        model (nn.Module): Built-in architecture with no instance forward override/hooks.

    Returns:
        dict: Recursive constructor options and per-module training flags.

    Raises:
        ValueError: If a custom module or unencoded runtime behavior is encountered.

    Notes:
        No weight arithmetic. Recursive subtree scans may cost O(L^2) for a deep
        L-module chain; the returned description uses O(H) metadata bytes.
    """
    for module in model.modules():
        if (
            module._forward_hooks
            or module._forward_pre_hooks
            or "forward" in module.__dict__
        ):
            raise ValueError(
                "models with custom forward hooks/instance forward are not encodable"
            )
    kind = type(model)
    options: dict = {}
    if kind is nn.Sequential:
        options["children"] = [
            [name, _architecture(child)] for name, child in model._modules.items()
        ]
    elif kind is nn.Linear:
        options = {
            "in_features": model.in_features,
            "out_features": model.out_features,
            "bias": model.bias is not None,
        }
    elif kind is nn.Conv2d:
        options = {
            name: getattr(model, name)
            for name in (
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "padding_mode",
            )
        }
        options["bias"] = model.bias is not None
    elif kind in (nn.BatchNorm1d, nn.BatchNorm2d):
        options = {
            name: getattr(model, name)
            for name in (
                "num_features",
                "eps",
                "momentum",
                "affine",
                "track_running_stats",
            )
        }
    elif kind is nn.ReLU:
        options = {"inplace": model.inplace}
    elif kind is nn.Flatten:
        options = {"start_dim": model.start_dim, "end_dim": model.end_dim}
    elif kind is nn.Dropout:
        options = {"p": model.p, "inplace": model.inplace}
    elif kind is nn.GELU:
        options = {"approximate": model.approximate}
    elif kind is nn.MaxPool2d:
        options = {
            name: getattr(model, name)
            for name in (
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "return_indices",
                "ceil_mode",
            )
        }
    elif kind is nn.AdaptiveAvgPool2d:
        options = {"output_size": model.output_size}
    elif kind not in (nn.Identity, nn.Sigmoid, nn.Tanh):
        raise ValueError(
            f"unsupported architecture: {kind.__name__}; no free external decoder assumed"
        )
    return {"type": kind.__name__, "options": options, "training": model.training}


def _build_architecture(description: dict) -> nn.Module:
    r"""Construct a fresh predictor from charged architecture metadata.

    Args:
        description (dict): Recursive whitelist constructor description.

    Returns:
        nn.Module: Fresh structure, with parameterized modules built on meta device.

    Raises:
        ValueError: If a constructor tag is unknown.

    Notes:
        O(L + H) metadata work. Meta construction avoids random initialization and
        dense parameter allocation until decoded state tensors are assigned.
    """
    constructors = {
        cls.__name__: cls
        for cls in (
            nn.Linear,
            nn.Conv2d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.ReLU,
            nn.Flatten,
            nn.Dropout,
            nn.GELU,
            nn.MaxPool2d,
            nn.AdaptiveAvgPool2d,
            nn.Identity,
            nn.Sigmoid,
            nn.Tanh,
        )
    }
    kind = description["type"]
    options = dict(description["options"])
    if kind == "Sequential":
        result = nn.Sequential(
            OrderedDict(
                (name, _build_architecture(child))
                for name, child in options["children"]
            )
        )
    elif kind in constructors:
        # Construct weights on meta so decoding neither samples an initialization
        # nor allocates a second full parameter set before loading decoded data.
        if kind in ("Linear", "Conv2d", "BatchNorm1d", "BatchNorm2d"):
            options["device"] = "meta"
        result = constructors[kind](**options)
    else:
        raise ValueError(f"unknown architecture tag: {kind}")
    result.training = bool(description["training"])
    return result


def _structure_only(description):
    r"""Remove mode flags when comparing transmitted and factory architectures.

    Args:
        description (object): Nested dictionary/list/scalar constructor metadata.

    Returns:
        object: Recursively copied metadata without ``training`` entries.

    Notes:
        O(H) work and auxiliary storage; this does not inspect state tensor values.
    """
    if isinstance(description, dict):
        return {
            key: _structure_only(value)
            for key, value in description.items()
            if key != "training"
        }
    if isinstance(description, (tuple, list)):
        return [_structure_only(value) for value in description]
    return description


def _raw(tensor: Tensor) -> bytes:
    # Serialization is an explicit CPU/byte boundary; no NumPy computation and
    # no hidden dtype conversion (including bfloat16 and scalar buffers).
    r"""Materialize exact tensor bytes at an explicit CPU serialization boundary.

    Args:
        tensor (Tensor): Strided state tensor; dtype is not changed.

    Returns:
        bytes: Native little-endian contiguous representation.

    Notes:
        O(B) byte traversal and copies for B bytes. CPU transfer, uint8 conversion
        and ``tolist`` also create O(B) Python integer objects. Autograd is detached.
    """
    return bytes(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()
    )


def _from_raw(payload: bytes, dtype: torch.dtype, shape: list[int]) -> Tensor:
    r"""Reconstruct a tensor from exact byte count, dtype and shape.

    Args:
        payload (bytes): Dense element bytes.
        dtype (torch.dtype): Explicit element dtype.
        shape (list[int]): Decoded dimensions.

    Returns:
        Tensor: Independent CPU tensor with the declared shape and dtype.

    Raises:
        ValueError: If byte count disagrees with dtype/shape.

    Notes:
        O(B) bytearray/copy work and storage; frombuffer is cloned to own memory.
    """
    size = math.prod(shape)
    if len(payload) != size * torch.empty((), dtype=dtype).element_size():
        raise ValueError("tensor byte count disagrees with dtype/shape")
    if not payload:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(bytearray(payload), dtype=dtype).clone().reshape(shape)


def _pack(values: Tensor, bits: int) -> bytes:
    r"""Pack fixed-width unsigned integer codes, including zero end padding.

    Args:
        values (Tensor): Integer-valued codes, copied to CPU and flattened.
        bits (int): Declared fixed width of each code.

    Returns:
        bytes: ceil(count * bits / 8) packed bytes.

    Raises:
        ValueError: If a code lies outside the declared alphabet.

    Notes:
        There are O(count + output bytes) Python iterations, but the current
        accumulator retains consumed bits and grows to the whole message. Python
        big-integer shifts may therefore take quadratic total bit-work, with
        O(message bits) accumulator storage. This is a correctness-first codec,
        not a promised linear-time production implementation.
    """
    output = bytearray()
    accumulator = filled = 0
    for value in values.detach().cpu().reshape(-1).tolist():
        value = int(value)
        if not 0 <= value < (1 << bits):
            raise ValueError("integer code exceeds declared bit width")
        accumulator = (accumulator << bits) | value
        filled += bits
        while filled >= 8:
            filled -= 8
            output.append((accumulator >> filled) & 255)
    if filled:
        output.append((accumulator << (8 - filled)) & 255)
    return bytes(output)


def _unpack(payload: bytes, count: int, bits: int) -> Tensor:
    r"""Validate padding and recover unsigned indices from packed bytes.

    Args:
        payload (bytes): Canonically padded fixed-width integer message.
        count (int): Number of expected codes.
        bits (int): Bit width per code.

    Returns:
        LongTensor: CPU indices [count].

    Raises:
        ValueError: If payload size or zero padding is inconsistent.

    Notes:
        The current Python accumulator retains consumed bytes: O(message bytes +
        count) loop iterations are not a linear bit-complexity guarantee. Growing
        big-integer shifts can take quadratic bit-work; output lists use O(count).
    """
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError("packed integer byte count is inconsistent")
    padding = (-count * bits) % 8
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ValueError("noncanonical nonzero padding bits")
    result: list[int] = []
    accumulator = filled = 0
    for value in payload:
        accumulator = (accumulator << 8) | value
        filled += 8
        while filled >= bits and len(result) < count:
            filled -= bits
            result.append((accumulator >> filled) & ((1 << bits) - 1))
    return torch.tensor(result, dtype=torch.int64)


def _tensor_message(
    name: str, tensor: Tensor, encoding: str, bits: int, rank: int
) -> tuple[dict, list[bytes]]:
    r"""Choose a tensor coding mode and emit charged metadata plus byte chunks.

    Args:
        name (str): State-dictionary tensor name.
        tensor (Tensor): Finite supported strided state tensor.
        encoding (str): Dense, sparse, low_rank or quantized strategy.
        bits (int): Fixed index width for quantization.
        rank (int): Maximum stored matrix factor rank.

    Returns:
        tuple[dict, list[bytes]]: Shape/dtype/mode/part sizes and actual payload parts.

    Raises:
        ValueError: If values, dtype, layout or SVD factors are unsupported.

    Notes:
        Dense is O(N) byte work; sparse adds an N-bit mask and nonzero values;
        quantization adds O(N) floating/index intermediates and bit packing.
        Economy SVD costs O(m*n*min(m,n)), even when the stored rank is small.
    """
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _DTYPES or tensor.layout != torch.strided:
        raise ValueError(f"unsupported tensor type/layout: {name}")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError("model code requires finite state tensors")
    record = {"name": name, "dtype": dtype, "shape": list(tensor.shape)}
    if encoding == "sparse" and tensor.is_floating_point():
        mask = tensor.reshape(-1) != 0
        record.update(encoding="sparse", nonzero=int(mask.sum()))
        parts = [_pack(mask, 1), _raw(tensor.reshape(-1)[mask])]
    elif encoding == "low_rank" and tensor.ndim == 2 and name.endswith("weight"):
        if tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError("low-rank code supports float32/float64 weight matrices")
        k = min(rank, *tensor.shape)
        u, s, vh = torch.linalg.svd(tensor.detach(), full_matrices=False)
        if (
            not torch.isfinite(u[:, :k] * s[:k]).all()
            or not torch.isfinite(vh[:k]).all()
        ):
            raise ValueError("SVD factors overflowed")
        record.update(encoding="low_rank", rank=k)
        parts = [_raw(u[:, :k] * s[:k]), _raw(vh[:k])]
    elif encoding == "quantized" and tensor.is_floating_point() and tensor.numel():
        quantized = uniform_quantize(tensor, bits)
        record.update(
            encoding="quantized",
            bits=bits,
            minimum=quantized.minimum,
            scale=quantized.scale,
        )
        parts = [_pack(quantized.codes, bits)]
    else:
        record.update(encoding="dense")
        parts = [_raw(tensor)]
    record["part_lengths"] = [len(part) for part in parts]
    return record, parts


def _tensor_decode(record: dict, parts: list[bytes]) -> Tensor:
    r"""Reconstruct one charged tensor message and validate mode-specific metadata.

    Args:
        record (dict): Shape, dtype, coding mode and side information.
        parts (list[bytes]): Exactly the chunks declared by the tensor record.

    Returns:
        Tensor: CPU reconstruction with original dtype and shape.

    Raises:
        ValueError: If shape, bit width, rank, scale, mask or chunk count is invalid.

    Notes:
        Dense/sparse/quantized paths allocate O(N) tensor storage, plus byte/index
        intermediates. Low-rank reconstruction multiplies [m,r]@[r,n], costing
        O(m*n*r) time and O(m*n + r*(m+n)) storage. Bit-unpack caveats still apply.
    """
    dtype = _DTYPES[record["dtype"]]
    shape = record["shape"]
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in shape):
        raise ValueError("invalid tensor shape")
    size = math.prod(shape)
    encoding = record["encoding"]
    if encoding == "dense" and len(parts) == 1:
        return _from_raw(parts[0], dtype, shape)
    if encoding == "sparse" and len(parts) == 2:
        mask = _unpack(parts[0], size, 1).bool()
        count = int(mask.sum())
        if count != record["nonzero"]:
            raise ValueError("sparse count disagrees with mask")
        values = _from_raw(parts[1], dtype, [count])
        decoded = torch.zeros(size, dtype=dtype)
        decoded[mask] = values
        return decoded.reshape(shape)
    if encoding == "low_rank" and len(parts) == 2 and len(shape) == 2:
        rank = record["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank <= min(shape)
        ):
            raise ValueError("invalid low-rank metadata")
        left = _from_raw(parts[0], dtype, [shape[0], rank])
        right = _from_raw(parts[1], dtype, [rank, shape[1]])
        return left @ right
    if encoding == "quantized" and len(parts) == 1:
        bits = record["bits"]
        if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
            raise ValueError("invalid quantization bit width")
        if (
            not math.isfinite(record["minimum"])
            or not math.isfinite(record["scale"])
            or record["scale"] <= 0
        ):
            raise ValueError("invalid quantization range")
        values = _unpack(parts[0], size, bits).to(torch.float64)
        return (values * record["scale"] + record["minimum"]).to(dtype).reshape(shape)
    raise ValueError("unknown tensor encoding or part count")


class BinaryModelCodec(ModelCodec):
    r"""Encode supported built-in PyTorch models in a versioned binary language.

    ``encoding`` selects dense values, sparse masks+values, matrix SVD factors,
    or packed affine quantization. Architecture, names, dtypes, shapes, frozen
    flags, padding, factors/scales and header lengths are all charged. Shared
    parameters/storage and custom architectures are rejected, not misdescribed.
    Dense/sparse code is lossless numerically; sparse code canonicalizes -0.
    Low-rank and quantized codes are explicitly lossy. The format is primarily
    a correctness reference, not a competitive production compressor.

    Args:
        encoding (str): ``dense``, ``sparse``, ``low_rank`` or ``quantized``.
        transform (ModelTransform, optional): Explicit non-mutating transform before coding.
        bits (int): Quantized integer width between one and sixteen.
        rank (int): Maximum matrix factor rank in low-rank coding.

    Notes:
        The format is a code language, not a claim of entropy-optimal compression.
        The default decoder restores CPU tensors; caller-visible relocation is
        explicit. Fixed decoder implementation is shared; per-model schema is charged.
    """

    def __init__(
        self,
        encoding: str = "dense",
        *,
        transform: ModelTransform | None = None,
        bits: int = 8,
        rank: int = 1,
    ):
        r"""Validate strategy settings and store components without reading a model.

        Args:
            encoding (str): Dense, sparse, low_rank or quantized.
            transform (ModelTransform, optional): Transformation applied explicitly by encode.
            bits (int): Quantized index width in [1, 16].
            rank (int): Nonnegative maximum stored rank.

        Raises:
            ValueError: If strategy, width or rank is invalid.

        Notes:
            O(1) storage/work; this object stores no fitted codebook or model state.
        """
        if encoding not in ("dense", "sparse", "low_rank", "quantized"):
            raise ValueError("unknown encoding")
        if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
            raise ValueError("bits must be in [1, 16]")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.encoding = encoding
        self.transform = transform
        self.bits = bits
        self.rank = rank

    def encode(self, model: nn.Module) -> EncodedModel:
        r"""Encode architecture and complete state into a physical binary message.

        Args:
            model (nn.Module): Supported built-in model with nonaliased state storage.

        Returns:
            EncodedModel: Magic/version, 8-byte header length, JSON metadata and tensor chunks.

        Raises:
            ValueError: If architecture, state, aliasing or host byte order is unsupported.

        Notes:
            Includes optional transform cost and per-tensor encoding work. Metadata,
            names, shapes, modes, masks/factors/scales and padding are all charged.
            Byte chunks and final concatenation may coexist, giving O(message bytes)
            extra storage beyond any transformed model, SVD and Python-list costs.
        """
        if sys.byteorder != "little":
            raise ValueError("format v1 supports little-endian hosts")
        if self.transform is not None:
            model = self.transform.apply(model)
        architecture = _architecture(model)
        parameters = list(model.named_parameters(remove_duplicate=False))
        if len({id(parameter) for _, parameter in parameters}) != len(parameters):
            raise ValueError("tied parameters require an alias-aware codec")
        state = model.state_dict()
        storages = [
            (str(t.device), t.untyped_storage().data_ptr())
            for t in state.values()
            if t.numel()
        ]
        if len(set(storages)) != len(storages):
            raise ValueError("shared storage requires an alias-aware codec")
        records: list[dict] = []
        chunks: list[bytes] = []
        for name, tensor in state.items():
            record, parts = _tensor_message(
                name, tensor, self.encoding, self.bits, self.rank
            )
            records.append(record)
            chunks.extend(parts)
        metadata = {
            "version": 1,
            "architecture": architecture,
            "state": records,
            "requires_grad": {
                name: parameter.requires_grad for name, parameter in parameters
            },
        }
        header = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload = _MAGIC + struct.pack(">Q", len(header)) + header + b"".join(chunks)
        return EncodedModel(payload)

    def decode(
        self,
        encoded: EncodedModel,
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> nn.Module:
        r"""Parse the complete message and rebuild the transmitted predictor.

        Args:
            encoded (EncodedModel): Versioned physical message.
            model_factory (callable, optional): Must agree with the charged schema; it
                cannot smuggle in a custom uncharged predictor.

        Returns:
            nn.Module: Fresh CPU model with decoded dtypes, state and frozen/mode flags.

        Raises:
            ValueError: If headers, chunks, schema or state assignment are inconsistent.

        Notes:
            Parsing/copying costs O(message bytes) besides tensor-specific unpack/SVD
            reconstruction costs. Peak storage includes message chunks and decoded
            tensors. No pickle is executed; this is not a malicious-input resource sandbox.
        """
        if sys.byteorder != "little":
            raise ValueError("format v1 supports little-endian hosts")
        payload = encoded.payload
        if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + 8:
            raise ValueError("invalid model-code magic/version")
        header_length = struct.unpack(">Q", payload[len(_MAGIC) : len(_MAGIC) + 8])[0]
        start = len(_MAGIC) + 8
        end = start + header_length
        if end > len(payload):
            raise ValueError("truncated model-code header")
        try:
            metadata = json.loads(payload[start:end])
            if metadata["version"] != 1:
                raise ValueError("unsupported code version")
            state: dict[str, Tensor] = {}
            offset = end
            for record in metadata["state"]:
                if record["name"] in state:
                    raise ValueError("duplicate state tensor")
                parts: list[bytes] = []
                for length in record["part_lengths"]:
                    if (
                        isinstance(length, bool)
                        or not isinstance(length, int)
                        or length < 0
                        or offset + length > len(payload)
                    ):
                        raise ValueError("truncated/invalid tensor payload")
                    parts.append(payload[offset : offset + length])
                    offset += length
                state[record["name"]] = _tensor_decode(record, parts)
            if offset != len(payload):
                raise ValueError("trailing bytes are not part of a model code")
            result = _build_architecture(metadata["architecture"])
            if model_factory is not None:
                supplied = model_factory()
                # A factory is a convenience, not an escape hatch for a free
                # custom architecture. Decoding uses the transmitted schema.
                if _structure_only(_architecture(supplied)) != _structure_only(
                    metadata["architecture"]
                ):
                    raise ValueError(
                        "model_factory disagrees with encoded architecture"
                    )
            result.load_state_dict(state, strict=True, assign=True)
            for name, parameter in result.named_parameters():
                parameter.requires_grad_(bool(metadata["requires_grad"][name]))
            return result
        except (KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid model code: {error}") from error
