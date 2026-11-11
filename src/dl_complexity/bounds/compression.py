"""Arora et al. (2018), Theorem 2.1, explicit confidence instantiation.

Source: https://proceedings.mlr.press/v80/arora18b/arora18b.pdf, Sec. 2;
proof: https://proceedings.mlr.press/v80/arora18b/arora18b-supp.pdf, A.1.
This is NOT the noise-stability bound of their Theorem 4.1.

Appendix A.1 gives Pr[R(g)-Rhat(g)>=t] <= exp(-2*n*t**2)
and counts at most r**q compressed predictors. Applying the same proof with
an explicit delta gives t=sqrt((q*ln(r)+ln(1/delta))/(2*n)). This constant
and arbitrary-delta parameterization are an explicit instantiation of that
proof, not a claim that the main paper prints this exact numerical formula.

Factor-two convention: the paper's Definition 1 uses coordinatewise output
error gamma, whereas its margin is f_y - max_{j!=y} f_j. Such a perturbation
can change a margin by 2*gamma. We require error <= margin_threshold/2 rather
than silently copying the apparent factor-two mismatch in the printed proof.
"""

from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch import Tensor, nn

from .._validation import positive_int, scalar
from ..errors import InputError
from ..results import (
    ComplexityResult,
    Provenance,
    Quantity,
    RequirementReport,
    RequirementStatus,
)
from .base import GeneralizationBoundEstimator
from .inputs import AroraCompressionInputs


def _validate(inputs: AroraCompressionInputs) -> None:
    r"""Validate the numerical compression certificate and explicit declarations.

    Args:
        inputs (AroraCompressionInputs): Scalar sample statistics and finite q/r budget.

    Raises:
        InputError: If counts, margin, distortion or declaration types are invalid.

    Notes:
        O(1) scalar work, including scalar finite/range checks and accelerator
        synchronization. It checks distortion <= margin/2, not independence/iid.
    """
    positive_int(inputs.n_observations, "n_observations")
    positive_int(inputs.values_per_parameter, "values_per_parameter")
    q = inputs.parameter_count
    if isinstance(q, bool) or not isinstance(q, int) or q < 0:
        raise InputError("parameter_count must be a nonnegative integer")
    margin = scalar(inputs.margin, "margin")
    if not bool(margin > 0):
        raise InputError("margin must be positive")
    empirical = scalar(inputs.empirical_margin_loss, "empirical_margin_loss")
    distortion = scalar(inputs.max_logit_error, "max_logit_error")
    if not bool((empirical >= 0) & (empirical <= 1)):
        raise InputError("empirical margin loss must be in [0, 1]")
    if not bool(distortion >= 0):
        raise InputError("max_logit_error must be nonnegative")
    if bool(distortion > margin.to(distortion) / 2):
        raise InputError("Arora certificate fails: max_logit_error must be <= margin/2")
    if not isinstance(inputs.helper_is_data_independent, bool) or not isinstance(
        inputs.alphabet_is_data_independent, bool
    ):
        raise InputError("independence declarations must be explicit booleans")


class AroraCompressionBound(GeneralizationBoundEstimator[AroraCompressionInputs]):
    """Upper-bound risk of compressed g using Theorem 2.1's proof.

    Assumes iid samples and a data-independent decoder/alphabet/helper string.
    The input certificate permits the empirical gamma-margin loss of original
    f to upper-bound empirical classification error of compressed g. The raw
    result may exceed one; it is never clipped or transferred to original f.
    Selecting q/r/helper after seeing data requires extra accounting, not just
    setting an independence flag. No layer-cushion/stability theorem is implied.
    """

    input_type = AroraCompressionInputs

    def __init__(self, delta: float = 0.05):
        r"""Store explicit confidence and the supported theorem convention.

        Args:
            delta (float): Failure probability in (0, 1).

        Raises:
            InputError: If delta is not finite and strictly between zero and one.

        Notes:
            O(1); no training, compression, sample evaluation or geometric analysis.
        """
        if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta < 1:
            raise InputError("delta must be finite and in (0, 1)")
        super().__init__(
            delta=delta,
            theorem="Arora2018_Theorem2.1_explicit_delta",
            margin_convention="coordinate_distortion_at_most_margin_over_two",
        )
        self.delta = delta

    def _check_inputs(self, inputs: AroraCompressionInputs) -> RequirementReport:
        r"""Check a scalar certificate and require explicit independent-family declarations.

        Args:
            inputs (AroraCompressionInputs): Full-sample certificate or user-supplied summary.

        Returns:
            RequirementReport: READY or missing declarations, with unproved assumptions.

        Raises:
            InputError: If scalar certificate arithmetic/metadata are invalid.

        Notes:
            O(1), no forward pass. Unlike purely shape checks, this method does validate
            supplied scalar values; booleans never prove data independence.
        """
        _validate(inputs)
        missing = []
        if not inputs.helper_is_data_independent:
            missing.append("data-independent decoder/helper-string declaration")
        if not inputs.alphabet_is_data_independent:
            missing.append(
                "data-independent discrete alphabet and q/r budget declaration"
            )
        return RequirementReport(
            RequirementStatus.MISSING_INPUTS if missing else RequirementStatus.READY,
            missing=tuple(missing),
            assumptions=(
                "Independence is user-declared, not inferred from a checkpoint",
                "Certificate covers every training observation",
            ),
        )

    def _estimate(self, inputs: AroraCompressionInputs) -> ComplexityResult:
        r"""Evaluate the explicit-delta instantiation of the Arora Theorem 2.1 proof.

        Args:
            inputs (AroraCompressionInputs): Empirical gamma-margin loss and finite q/r class.

        Returns:
            ComplexityResult: Raw compressed-predictor risk upper bound and components.

        Notes:
            bound = empirical_margin_loss + sqrt((q*log(r)-log(delta))/(2*n)).
            O(1) arithmetic/storage. No clipping at one, no conversion to code units,
            and no transfer to the original predictor f is silently performed.
        """
        empirical = scalar(inputs.empirical_margin_loss, "empirical_margin_loss")
        log_cardinality = inputs.parameter_count * math.log(inputs.values_per_parameter)
        # Keep the supplied floating dtype/device; this bound is not a log code.
        penalty = empirical.new_tensor(
            (log_cardinality - math.log(self.delta)) / (2 * inputs.n_observations)
        ).sqrt()
        quantities = {
            "risk_upper_bound": Quantity(empirical + penalty, "probability"),
            "empirical_margin_loss": Quantity(empirical, "probability"),
            "complexity_penalty": Quantity(penalty, "probability"),
            "log_cardinality_upper_bound": Quantity(
                empirical.new_tensor(log_cardinality), "nats"
            ),
            "max_logit_error": Quantity(
                scalar(inputs.max_logit_error).to(empirical), "dimensionless"
            ),
        }
        return ComplexityResult(
            method="AroraCompressionBound",
            kind="compressed_predictor_risk_bound",
            quantities=quantities,
            primary_name="risk_upper_bound",
            assumptions=(
                "Arora et al. 2018 Theorem 2.1, explicit delta from Supplement A.1 proof",
                "iid observations; fixed q/r alphabet, decoder and helper independent of sample",
                "max coordinate logit distortion <= gamma/2 on entire sample",
                "guarantee concerns compressed g, NOT original f",
                "does not implement Theorem 4.1 noise-stability compression or geometric estimates",
                f"compression certificate: {inputs.certificate_kind}",
            ),
            provenance=inputs.provenance,
        )


def arora_inputs_from_logits(
    original_logits: Tensor,
    compressed_logits: Tensor,
    targets: Tensor,
    *,
    parameter_count: int,
    values_per_parameter: int,
    margin: float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    compressed_state_id: str | None = None,
    provenance: Provenance | None = None,
) -> AroraCompressionInputs:
    r"""Measure a full-sample certificate, not a sampled/subset max-error proxy.

    Logits must be [n, classes], finite and on the same device/dtype. Targets
    index classes; ties and margins equal to the threshold count as margin
    errors. Non-differentiable certificate statistics do not retain a graph.
    This function cannot establish whether the declared class/helper was fixed
    before these training observations were seen.

    Args:
        original_logits (Tensor): Original predictor outputs [n, classes >= 2].
        compressed_logits (Tensor): Matching decoded predictor outputs.
        targets (LongTensor): Class-index labels [n].
        parameter_count (int): Independently fixed upper coefficient budget q.
        values_per_parameter (int): Independently fixed alphabet bound r.
        margin (float): Positive original-predictor margin threshold gamma.
        helper_is_data_independent (bool): Explicit fixed-helper declaration.
        alphabet_is_data_independent (bool): Explicit fixed q/r/alphabet declaration.
        compressed_state_id (str, optional): Declared identity of g.
        provenance (Provenance, optional): User-declared state/data identities.

    Returns:
        AroraCompressionInputs: Empirical margin loss and maximum coordinate error.

    Raises:
        InputError: If inputs are inconsistent or distortion exceeds gamma/2.

    Notes:
        O(n*C) reductions and O(n*C) competitor/difference temporaries; no_grad.
        A subset of logits is not a full-sample certificate. Ties count as errors.
    """
    if not isinstance(original_logits, Tensor) or not isinstance(
        compressed_logits, Tensor
    ):
        raise InputError("logits must be PyTorch tensors")
    if (
        original_logits.ndim != 2
        or original_logits.shape != compressed_logits.shape
        or original_logits.shape[0] == 0
        or original_logits.shape[1] < 2
    ):
        raise InputError(
            "both logit tensors must have the same nonempty [n, classes>=2] shape"
        )
    if (
        not original_logits.is_floating_point()
        or not compressed_logits.is_floating_point()
        or original_logits.dtype != compressed_logits.dtype
        or original_logits.device != compressed_logits.device
    ):
        raise InputError("logits must share a floating dtype and device")
    if (
        not torch.isfinite(original_logits).all()
        or not torch.isfinite(compressed_logits).all()
    ):
        raise InputError("finite logits are required")
    if (
        not isinstance(targets, Tensor)
        or targets.ndim != 1
        or len(targets) != len(original_logits)
        or targets.dtype not in (torch.int32, torch.int64)
        or targets.device != original_logits.device
    ):
        raise InputError(
            "targets must be a matching integer class-index vector on the logits' device"
        )
    if not bool(((targets >= 0) & (targets < original_logits.shape[1])).all()):
        raise InputError("target class index out of range")
    with torch.no_grad():
        indices = targets.to(torch.int64).unsqueeze(1)
        true_score = original_logits.gather(1, indices).squeeze(1)
        competitors = original_logits.detach().clone().scatter_(1, indices, -torch.inf)
        margins = true_score - competitors.max(dim=1).values
        empirical = (margins <= margin).to(original_logits.dtype).mean()
        distortion = (original_logits - compressed_logits).abs().max()
    result = AroraCompressionInputs(
        empirical,
        len(targets),
        parameter_count,
        values_per_parameter,
        margin,
        distortion,
        helper_is_data_independent,
        alphabet_is_data_independent,
        "measured_full_training_logits",
        compressed_state_id,
        provenance,
    )
    _validate(result)
    return result


def prepare_arora_compression(
    original: nn.Module,
    compressed: nn.Module,
    data,
    *,
    parameter_count: int,
    values_per_parameter: int,
    margin: float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    compressed_state_id: str | None = None,
) -> AroraCompressionInputs:
    r"""Explicitly evaluate a certificate on all batches using isolated copies.

    Scope: tensor inputs, class-index targets, deterministic eval logits for
    ordinary small MLP/CNN models. This is preparation, never implicit training
    or telemetry. Caller keeps a re-iterable source if other methods need it.
    q/r describe a fixed discrete predictor family; codec bytes do not infer it.

    Args:
        original (nn.Module): Original predictor f, copied before eval.
        compressed (nn.Module): Decoded candidate g, also copied.
        data (DataSource): Entire training sample as tensor/class-index batches.
        parameter_count (int): Declared fixed finite-coordinate budget q.
        values_per_parameter (int): Declared fixed alphabet upper bound r.
        margin (float): Positive margin threshold.
        helper_is_data_independent (bool): User-declared helper independence.
        alphabet_is_data_independent (bool): User-declared alphabet/budget independence.
        compressed_state_id (str, optional): Identity of g, not f.

    Returns:
        AroraCompressionInputs: Measured sample certificate with explicit assumptions.

    Raises:
        InputError: If batches or sample/certificate constraints are invalid.

    Notes:
        Two model copies O(P_f+P_g), two full forward passes, and stored logits
        O(n*C) plus concatenation/difference temporaries. The implementation does
        not reduce the maximum online; it materializes all sample logits.
        CPU/model-parameter CUDA RNG is scoped; arbitrary external side effects
        or RNG on devices absent from those parameters are not universally isolated.
    """
    first, second = deepcopy(original).eval(), deepcopy(compressed).eval()
    original_rows, compressed_rows, target_rows = [], [], []
    cuda_devices = sorted(
        {
            p.device.index
            for m in (first, second)
            for p in m.parameters()
            if p.device.type == "cuda"
        }
    )
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        for batch in data:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise InputError("each batch must be (inputs, class_indices)")
            x, y = batch
            if not isinstance(x, Tensor) or not isinstance(y, Tensor):
                raise InputError(
                    "Arora preparation currently supports tensor inputs/targets"
                )
            original_rows.append(first(x))
            compressed_rows.append(second(x))
            target_rows.append(y)
    if not target_rows:
        raise InputError("empty training sample")
    targets = torch.cat(target_rows)
    declared_n = getattr(data, "n_observations", None)
    if declared_n is not None and declared_n != len(targets):
        raise InputError("declared observation count disagrees with certificate sample")
    return arora_inputs_from_logits(
        torch.cat(original_rows),
        torch.cat(compressed_rows),
        targets,
        parameter_count=parameter_count,
        values_per_parameter=values_per_parameter,
        margin=margin,
        helper_is_data_independent=helper_is_data_independent,
        alphabet_is_data_independent=alphabet_is_data_independent,
        compressed_state_id=compressed_state_id,
        provenance=Provenance(
            state_id=compressed_state_id, data_id=getattr(data, "data_id", None)
        ),
    )


def prepare_arora_projection(
    original: nn.Module,
    projected,
    data,
    *,
    margin: float,
    helper_is_data_independent: bool,
    alphabet_is_data_independent: bool,
    compressed_state_id: str | None = None,
) -> AroraCompressionInputs:
    r"""Certify an actual Matrix-Project artifact, deriving q/r from its code.

    The ENTIRE predictor schema is revalidated (no hooks, overrides, buffers,
    structural changes or aliases), then g is freshly decoded from its bound
    architecture and finite indices. Mutable instance forward is never used
    for compressed logits. Independence of helper/alphabet remains an explicit
    user declaration, not something logits can prove.

    Args:
        original (nn.Module): Original predictor f.
        projected (ProjectedModel): Actual finite-index Matrix-Project artifact.
        data (DataSource): Entire training sample.
        margin (float): Positive margin threshold.
        helper_is_data_independent (bool): Explicit helper independence declaration.
        alphabet_is_data_independent (bool): Explicit alphabet/q/r independence declaration.
        compressed_state_id (str, optional): Declared identity of decoded g.

    Returns:
        AroraCompressionInputs: Certificate bound to decoded indices and measured logits.

    Raises:
        InputError: If the projection schema/state or sample certificate is invalid.

    Notes:
        Adds ProjectedModel.decode costs to two full-sample forward passes and
        O(n*C) retained logits. q/r come from finite indices, never codec bytes.
        Independence remains an assumption; full noise-stability Theorem 4.1,
        layer cushions, rho_delta and a geometry-based budget are not implemented.
    """
    from ..compression.arora import ProjectedModel

    if not isinstance(projected, ProjectedModel) or not projected.projections:
        raise InputError("expected a nonempty ProjectedModel artifact")
    try:
        decoded = projected.decode()
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise InputError(f"invalid Arora finite predictor: {error}") from error
    certificate = prepare_arora_compression(
        original,
        decoded,
        data,
        parameter_count=projected.parameter_count,
        values_per_parameter=projected.values_per_parameter,
        margin=margin,
        helper_is_data_independent=helper_is_data_independent,
        alphabet_is_data_independent=alphabet_is_data_independent,
        compressed_state_id=compressed_state_id,
    )
    return replace(
        certificate,
        certificate_kind="matrix_project_finite_indices_and_full_training_logits",
    )
