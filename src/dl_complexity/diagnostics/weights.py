"""Unique-parameter statistics; matrix spectral norms are not convolution norms."""

import torch

from ..errors import InputError
from ..results import (
    ComplexityResult,
    Provenance,
    Quantity,
    RequirementReport,
    RequirementStatus,
)
from .base import DiagnosticEstimator
from .inputs import WeightInputs


class WeightStatistics(DiagnosticEstimator):
    r"""Measure unique-tensor weight norms, sparsity and optional matrix spectral norms.

    Args:
        trainable_only (bool): Context route selects only requires_grad parameters.
        spectral (bool): Include matrix operator-2 norms for exactly 2D tensors.

    Notes:
        No primary scalar is forced. A flattened convolution filter's matrix norm
        would not automatically be the convolution operator norm.
    """

    input_type = WeightInputs
    uses_data = False

    def __init__(self, *, trainable_only=False, spectral=False):
        r"""Store tensor-selection and spectral-norm settings without reading weights."""
        super().__init__(trainable_only=trainable_only, spectral=spectral)
        self.trainable_only, self.spectral = trainable_only, spectral

    def _check_context(self, context, data):
        r"""Accept a context route without evaluating a model or consuming data."""
        if context is None:
            return super()._check_context(context, data)
        return RequirementReport(RequirementStatus.READY)

    def _prepare_context(self, context, data):
        r"""Select named unique parameters and attach a declared state identity.

        Notes:
            O(number of parameter objects) mapping work; tensors are not copied here.
        """
        return WeightInputs(
            {
                n: p
                for n, p in context.model.named_parameters()
                if not self.trainable_only or p.requires_grad
            },
            Provenance(state_id=context.state_id),
        )

    def _estimate(self, inputs):
        # Tied Parameter objects count once, including for direct-input mappings.
        r"""Validate tensors, deduplicate identity, then compute local and global statistics.

        Args:
            inputs (WeightInputs): Named floating weights on a common dtype/device.

        Returns:
            ComplexityResult: Per-tensor L2, optional 2D spectral norm, global L1/L2,
                count and fraction of exact zeros. There is no primary score.

        Raises:
            InputError: Empty selection, nonfloating/nonfinite weights or mixed global dtype/device.

        Notes:
            Concatenation, finite checks, zero counting and vector norms take O(P)
            work/storage. Spectral norms add dense singular-value work per matrix.
            Tensor identity is deduplicated; distinct objects sharing storage are not.
            Count is represented in the weight dtype, so low precision can round large P.
        """
        seen, weights = set(), []
        values = {}
        for name, tensor in inputs.weights.items():
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise InputError("Weight statistics expect floating tensors")
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            if not bool(torch.isfinite(tensor).all()):
                raise InputError("Weights must be finite")
            weights.append(tensor.reshape(-1))
            values[f"{name}.l2"] = Quantity(
                torch.linalg.vector_norm(tensor), "weight_norm"
            )
            if self.spectral and tensor.ndim == 2:
                values[f"{name}.spectral"] = Quantity(
                    torch.linalg.matrix_norm(tensor, ord=2), "weight_norm"
                )
        if not weights:
            raise InputError("No selected weights")
        if len({(w.device, w.dtype) for w in weights}) != 1:
            raise InputError(
                "Global norms require a common device/dtype; select tensors explicitly"
            )
        vector = torch.cat(weights)
        values.update(
            parameter_count=Quantity(vector.new_tensor(vector.numel()), "count"),
            l1=Quantity(vector.abs().sum(), "weight_norm"),
            l2=Quantity(torch.linalg.vector_norm(vector), "weight_norm"),
            sparsity=Quantity((vector == 0).to(vector.dtype).mean(), "dimensionless"),
        )
        return ComplexityResult(
            type(self).__name__,
            "weight_diagnostics",
            values,
            assumptions=("Unique tensor identity, not unique storage addresses",),
        )
