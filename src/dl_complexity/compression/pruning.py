"""Weight-only magnitude pruning; no sensitivity or fine-tuning is implied."""

from copy import deepcopy
import math

import torch
from torch import nn

from .base import ModelTransform


class MagnitudePruning(ModelTransform):
    """Set the smallest ``floor(fraction * N)`` matrix/kernel weights to zero.

    Selection is global across parameters with ndim >= 2; biases are untouched.
    Ties use stable parameter/flat-index order. This transformation does not by
    itself define a description length: use an explicit sparse codec afterward.
    """

    def __init__(self, fraction: float = 0.5):
        r"""Validate and store a weight-only transform configuration.

        Args:
            fraction: Finite global removal fraction in [0, 1].

        Raises:
            ValueError: If the configuration is invalid.

        Notes:
            O(1) time and storage; no model is read or fitted.
        """
        if (
            isinstance(fraction, bool)
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("fraction must be finite and in [0, 1]")
        self.fraction = fraction

    def apply(self, model: nn.Module) -> nn.Module:
        r"""Copy the model and zero globally smallest eligible weights with stable ties.

        Args:
            model (nn.Module): Finite weights; eligible ndim >= 2 parameters share a device.

        Returns:
            nn.Module: Independent model; biases and ineligible parameters are unchanged.

        Raises:
            ValueError: If eligible weights are nonfinite or span several devices.

        Notes:
            For P copied parameters and N eligible weights: O(P + N*log(N)) time,
            O(P + N) extra storage. A full stable argsort is used, not linear selection.
            Magnitudes are explicitly float64 for comparison; returned dtype/device
            are preserved. Discrete mask selection does not retain autograd.
        """
        result = deepcopy(model)
        parameters = [p for p in result.parameters() if p.ndim >= 2]
        if not parameters:
            return result
        devices = {p.device for p in parameters}
        if len(devices) != 1:
            raise ValueError("global pruning requires parameters on one device")
        magnitudes = torch.cat(
            [p.detach().abs().reshape(-1).to(torch.float64) for p in parameters]
        )
        if not torch.isfinite(magnitudes).all():
            raise ValueError("pruning requires finite weights")
        count = math.floor(self.fraction * magnitudes.numel())
        selected = torch.argsort(magnitudes, stable=True)[:count]
        mask = torch.ones(
            magnitudes.numel(), device=magnitudes.device, dtype=torch.bool
        )
        mask[selected] = False
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                size = parameter.numel()
                parameter.mul_(mask[offset : offset + size].reshape(parameter.shape))
                offset += size
        return result
