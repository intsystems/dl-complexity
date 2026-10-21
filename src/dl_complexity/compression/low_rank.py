"""Explicit weight-matrix SVD approximation, not data-aware compression."""

from copy import deepcopy

import torch
from torch import nn

from .base import ModelTransform


class LowRankApproximation(ModelTransform):
    """Truncate each two-dimensional weight matrix to at most ``rank``.

    Convolutional kernels and biases are not silently flattened. The returned
    network has the same forward architecture; a low-rank codec stores factors
    instead of dense reconstructions to obtain an actual model description.
    """

    def __init__(self, rank: int):
        r"""Validate and store a weight-only transform configuration.

        Args:
            rank: Nonnegative maximum rank for each two-dimensional matrix.

        Raises:
            ValueError: If the configuration is invalid.

        Notes:
            O(1) time and storage; no model is read or fitted.
        """
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        self.rank = rank

    def apply(self, model: nn.Module) -> nn.Module:
        r"""Copy the model and replace each matrix by its truncated SVD reconstruction.

        Args:
            model (nn.Module): Finite float32/float64 two-dimensional weight parameters.

        Returns:
            nn.Module: Same architecture with approximate dense matrices; kernels/biases stay.

        Raises:
            ValueError: If matrix dtype or finite-value checks are unsupported.

        Notes:
            For matrix m*n and d=min(m,n), economy SVD costs O(m*n*d); storage for U/S/Vh
            is O((m+n)*d). Reconstruction adds O(m*n*rank) work and O(m*n) storage.
            The model copy costs O(P); this is not a randomized or truncated-only SVD.
            ``no_grad`` makes this preparation nondifferentiable through the original.
        """
        result = deepcopy(model)
        with torch.no_grad():
            for parameter in result.parameters():
                if parameter.ndim != 2:
                    continue
                if parameter.dtype not in (torch.float32, torch.float64):
                    raise ValueError("SVD currently supports float32/float64 matrices")
                if not torch.isfinite(parameter).all():
                    raise ValueError("SVD requires finite weights")
                rank = min(self.rank, *parameter.shape)
                u, s, vh = torch.linalg.svd(parameter, full_matrices=False)
                approximation = (u[:, :rank] * s[:rank]) @ vh[:rank]
                if not torch.isfinite(approximation).all():
                    raise ValueError("SVD approximation overflowed")
                parameter.copy_(approximation)
        return result
