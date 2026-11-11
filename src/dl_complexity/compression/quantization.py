"""Weight-only affine uniform quantization with an explicit rounding rule."""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .base import ModelTransform


@dataclass(frozen=True)
class QuantizedTensor:
    r"""Integer codes plus the metadata required for deterministic decoding.

    Args:
        codes (LongTensor): Integer level indices, with the original tensor shape.
        minimum (float): Transmitted lower endpoint, not free side information.
        scale (float): Positive transmitted spacing between consecutive levels.
        bits (int): Declared level-index bit width.
        dtype (torch.dtype): Floating dtype restored by decoding.

    Notes:
        The wrapper is O(1); codes retain O(N) int64 storage, not a packed bitstream.
        Use BinaryModelCodec to measure physical packed length.
    """

    codes: Tensor
    minimum: float
    scale: float
    bits: int
    dtype: torch.dtype

    def dequantize(self) -> Tensor:
        r"""Decode on the codes' device and restore the original floating dtype.

        Returns:
            Tensor: codes * scale + minimum, cast to the original dtype on the codes' device.

        Notes:
            O(N) arithmetic/storage. Float64 intermediate arithmetic is deliberate;
            rounding/index selection in the encoder is not differentiated.
        """
        return (self.codes.to(torch.float64) * self.scale + self.minimum).to(self.dtype)


def uniform_quantize(tensor: Tensor, bits: int = 8) -> QuantizedTensor:
    r"""Quantize to 2**bits levels using min/max and nearest-even rounding.

    The range/scale are learned values, not free side information: a codec must
    transmit them. Float64 arithmetic is explicit here for range calculation;
    the decoded tensor returns to the original dtype and device.

    Args:
        tensor (Tensor): Nonempty finite floating tensor with N elements.
        bits (int): Number of bits per index, between one and sixteen.

    Returns:
        QuantizedTensor: Int64 indices plus min/scale/original-dtype metadata.

    Raises:
        ValueError: If bit width, input values or representable range are invalid.

    Notes:
        O(N) reductions and elementwise work, O(N) float64 and int64 intermediates.
        Constant tensors use scale=1 and zero codes. Scalar min/max extraction may
        synchronize an accelerator; the source tensor is detached, not mutated.
    """
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer in [1, 16]")
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError("quantization requires a nonempty floating tensor")
    values = tensor.detach().to(torch.float64)
    if not torch.isfinite(values).all():
        raise ValueError("quantization requires finite values")
    minimum = float(values.min())
    maximum = float(values.max())
    scale = (maximum - minimum) / ((1 << bits) - 1) if maximum != minimum else 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            "quantization range overflows/underflows the declared float64 arithmetic"
        )
    codes = (
        torch.round((values - minimum) / scale)
        .clamp_(0, (1 << bits) - 1)
        .to(torch.int64)
    )
    return QuantizedTensor(codes, minimum, scale, bits, tensor.dtype)


class UniformQuantization(ModelTransform):
    """Replace floating parameters by their decoded affine-grid values."""

    def __init__(self, bits: int = 8):
        r"""Validate and store a weight-only transform configuration.

        Args:
            bits: Integer number of quantization bits in [1, 16].

        Raises:
            ValueError: If the configuration is invalid.

        Notes:
            O(1) time and storage; no model is read or fitted.
        """
        if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
            raise ValueError("bits must be an integer in [1, 16]")
        self.bits = bits

    def apply(self, model: nn.Module) -> nn.Module:
        r"""Copy the model and quantize/dequantize its nonempty floating parameters.

        Args:
            model (nn.Module): Predictor whose parameters may use supported floating dtypes.

        Returns:
            nn.Module: Independent dense model with rounded parameter values.

        Notes:
            O(P) model copy and elementwise work; peak extra storage includes the copy
            plus float64/int64 intermediates for the largest tensor. Buffers are not
            transformed here. No QAT, activation calibration or fine-tuning is performed.
        """
        result = deepcopy(model)
        with torch.no_grad():
            for parameter in result.parameters():
                if parameter.is_floating_point() and parameter.numel():
                    parameter.copy_(uniform_quantize(parameter, self.bits).dequantize())
        return result
