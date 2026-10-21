"""Two-part MDL: L(model) + L(data | decoded model), with explicit units.

See Grünwald, A Tutorial Introduction to MDL, Sections 2.4--2.5.
The codec owns precision, architecture, side-information and decoder overhead.
"""

import math
from dataclasses import replace

import torch

from .._validation import positive_int, scalar
from ..errors import InputError
from ..results import ComplexityResult, Quantity, RequirementReport, RequirementStatus
from .base import CodeLengthEstimator
from .inputs import TwoPartInputs


class TwoPartMDL(CodeLengthEstimator[TwoPartInputs]):
    r"""Compute a two-part model-plus-data description length.

    Args:
        unit (str): Output logarithmic unit, ``nats`` or ``bits``.

    Notes:
        The inherited ``estimate`` consumes TwoPartInputs; preparation is explicitly
        separate. Scalar arithmetic costs O(1), whereas encoding and evaluating the
        decoded model generally dominate an end-to-end call.

    Examples:
        >>> inputs = TwoPartInputs(model_bits=8.0, data_nll=2.0, n=2)
        >>> TwoPartMDL(unit="bits").estimate(inputs=inputs).primary.unit
        'bits'
    """

    input_type = TwoPartInputs

    def __init__(self, *, unit: str = "nats"):
        r"""Configure output units without computing or encoding any data.

        Args:
            unit (str): ``nats`` or ``bits``.

        Raises:
            InputError: If the unit is not a logarithmic code unit.

        Notes:
            Time and additional memory are O(1).
        """
        if unit not in {"nats", "bits"}:
            raise InputError("A code length uses nats or bits")
        super().__init__(unit=unit)
        self.unit = unit

    def _check_inputs(self, inputs):
        r"""Validate scalar shapes, observation count and the declared code convention.

        Args:
            inputs (TwoPartInputs): Prepared model/data terms.

        Returns:
            RequirementReport: READY with the still user-declared coding assumptions.

        Raises:
            InputError: If n, scalar shapes or the convention are invalid.

        Notes:
            O(1) metadata checks; no forward pass, codec operation or tensor reduction.
        """
        positive_int(inputs.n, "n")
        if inputs.data_code_convention not in {"discrete", "quantized"}:
            raise InputError(
                "Declare a discrete or resolution-adjusted quantized data code"
            )
        for name in ("model_bits", "data_nll", "resolution_adjustment_nats"):
            value = getattr(inputs, name)
            if isinstance(value, torch.Tensor) and value.ndim != 0:
                raise InputError(f"{name} must be scalar")
        return RequirementReport(
            RequirementStatus.READY,
            assumptions=(
                "Codec completeness, resolution and decoded-predictor identity are user-declared",
            ),
        )

    def _estimate(self, inputs):
        r"""Add scalar model and decoded-data costs in a common logarithm base.

        Args:
            inputs (TwoPartInputs): Model bits, data nats and optional cell-volume term.

        Returns:
            ComplexityResult: Total, model, data and per-observation quantities.

        Raises:
            InputError: If terms are nonfinite, negative code costs or contradictory.

        Notes:
            L_total[nats] = model_bits * log(2) + data_nll + correction.
            Quantized density codes are labelled approximations, not exact bin integrals.
            Time/memory are O(1); supplied differentiable scalar tensors keep autograd.
        """
        n = positive_int(inputs.n, "n")
        model_bits = scalar(inputs.model_bits, "model_bits")
        data_nll = scalar(inputs.data_nll, "data_nll")
        correction = scalar(
            inputs.resolution_adjustment_nats, "resolution_adjustment_nats"
        )
        if bool(model_bits < 0):
            raise InputError("Model code cannot have negative length")
        if inputs.data_code_convention not in {"discrete", "quantized"}:
            raise InputError(
                "Declare a discrete or resolution-adjusted quantized data code"
            )
        if inputs.data_code_convention == "discrete" and bool(correction != 0):
            raise InputError("A discrete code does not use a resolution correction")
        data_code = data_nll + correction
        if bool(data_code < 0):
            raise InputError(
                "Negative density NLL is not a data code without a valid resolution correction"
            )
        # Conversion only changes the logarithm base; model/data semantics stay separate.
        model_code = model_bits * math.log(2)
        if self.unit == "bits":
            model_code, data_code = model_code / math.log(2), data_code / math.log(2)
        total = model_code + data_code
        return ComplexityResult(
            method=type(self).__name__,
            kind=(
                "two_part_code"
                if inputs.data_code_convention == "discrete"
                else "two_part_density_code_approximation"
            ),
            quantities={
                "total": Quantity(total, self.unit),
                "model": Quantity(model_code, self.unit),
                "data": Quantity(data_code, self.unit),
                "per_observation": Quantity(total / n, self.unit, "per_observation"),
            },
            primary_name="total",
            assumptions=(
                "Model code includes decoder side-information and overhead",
                "Data cost refers to the decoded predictor",
                f"Data code convention: {inputs.data_code_convention}",
            )
            + (
                (
                    "Density times cell volume approximates, rather than integrates, bin probability",
                )
                if inputs.data_code_convention == "quantized"
                else ()
            ),
        )


def prepare_two_part(
    context,
    *,
    data,
    codec,
    model_factory=None,
    data_code_convention="discrete",
    resolution_adjustment_nats=0.0,
):
    r"""Encode/decode explicitly, then evaluate the decoded model's likelihood.

    This is preparation, not a hidden operation of ``estimate``. Codec metadata
    is counted through actual payload bytes. Codec decoding is nondifferentiable.

    Args:
        context (ModelContext): Original model and explicit likelihood.
        data (DataSource): Observation batches evaluated after decoding.
        codec (ModelCodec): Encoder/decoder that charges all side information.
        model_factory (callable, optional): Optional schema-consistent factory.
        data_code_convention (str): Discrete code or quantized-density approximation.
        resolution_adjustment_nats (float or Tensor): Declared density correction.

    Returns:
        TwoPartInputs: Actual payload bits and decoded-predictor data NLL.

    Raises:
        InputError: If the likelihood or data conventions are invalid.
        ValueError: If the codec cannot represent or decode the model.

    Notes:
        Cost is codec encode + decode + a full likelihood evaluation. It retains a
        payload, decoded model and integration copies: O(message bytes + P + buffers)
        plus forward/autograd storage. Device is relocated, but mixed dtypes are
        preserved; derivatives do not cross discrete encode/decode.
    """
    from ..integrations.torch.likelihood import prepare_fit_statistics

    encoded = codec.encode(context.model)
    decoded = codec.decode(encoded, model_factory=model_factory)
    first = next(context.model.parameters(), None)
    if first is not None:
        # Device relocation is permitted, dtype homogenisation is not: a mixed-
        # precision description must still evaluate the predictor it encoded.
        decoded = decoded.to(device=first.device)
    decoded_context = replace(context, model=decoded, state_id=None)
    statistics = prepare_fit_statistics(decoded_context, data=data)
    return TwoPartInputs(
        encoded.length_bits,
        -statistics.log_likelihood,
        statistics.n,
        provenance=statistics.provenance,
        data_code_convention=data_code_convention,
        resolution_adjustment_nats=resolution_adjustment_nats,
    )
