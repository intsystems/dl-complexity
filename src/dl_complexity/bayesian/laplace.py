"""Laplace free energy with explicit selected dimension and total precision.

-log Z ~= -log L(theta_MAP) - log prior(theta_MAP)
          - d/2 log(2*pi) + 1/2 logdet(H_likelihood + H_prior).
Ritter et al. (2018), A Scalable Laplace Approximation for Neural Networks.
Exact isotropic addition here differs from their factor-wise damping Eq. 9.
"""

import math

import torch

from dl_complexity._validation import positive_int, scalar
from dl_complexity.bayesian._checks import scalar_metadata
from dl_complexity.bayesian.base import BayesianEstimator
from dl_complexity.bayesian.inputs import KFACFactors, KroneckerBlock, LaplaceTerms
from dl_complexity.errors import InputError
from dl_complexity.results import ComplexityResult, Quantity


def _psd_eigenvalues(factor, name):
    r"""Validate one symmetric PSD factor and return its eigenvalues.

    Args:
        factor (Tensor): Nonempty floating square matrix [m, m].
        name (str): Factor label for diagnostics.

    Returns:
        Tensor: Length-m eigenvalues of (factor + factor.T)/2, with only tiny
        negative roundoff within the numerical tolerance clipped to zero.

    Raises:
        InputError: If values are nonfinite, symmetry fails, or an eigenvalue
            is below -100*eps*max(1, max(abs(factor))).

    Notes:
        Numerical checks take O(m^2), eigvalsh O(m^3), and matrix/workspace
        storage O(m^2). Scalar tolerance extraction synchronizes accelerators.
        This is numerical validation, not a damping heuristic or a proof that
        empirical Fisher, GGN and Hessian are interchangeable.
    """
    if (
        not isinstance(factor, torch.Tensor)
        or factor.ndim != 2
        or factor.shape[0] != factor.shape[1]
        or factor.shape[0] == 0
    ):
        raise InputError(f"{name} must be a nonempty square Tensor")
    if not factor.is_floating_point() or not bool(torch.isfinite(factor).all()):
        raise InputError(f"{name} must contain finite floating values")
    tolerance = (
        100
        * torch.finfo(factor.dtype).eps
        * max(1.0, float(factor.detach().abs().max()))
    )
    if not torch.allclose(
        factor, factor.mT, atol=tolerance, rtol=100 * torch.finfo(factor.dtype).eps
    ):
        raise InputError(f"{name} must be symmetric")
    eigenvalues = torch.linalg.eigvalsh((factor + factor.mT) / 2)
    if bool((eigenvalues < -tolerance).any()):
        raise InputError(f"{name} must be positive semidefinite")
    # Clip only numerical negative roundoff within the documented PSD tolerance.
    return eigenvalues.clamp_min(0)


class KFACLaplace(BayesianEstimator[LaplaceTerms | KFACFactors]):
    r"""Evaluate a Laplace free energy from total precision or KFAC factors.

    Formula::

        negative_log_evidence = -log_likelihood - log_prior
            - dimension/2 * log(2*pi) + logdet_precision/2

    Notes:
        Inputs are either LaplaceTerms or KFACFactors. For Kronecker blocks,
        exact isotropic prior addition is used, rather than Ritter et al.'s
        factor-wise damping approximation. The reference point must be MAP to
        interpret the result as evidence; the reducer does not establish MAP.
        It neither collects general-network factors nor fits the model.
    """

    input_type = (LaplaceTerms, KFACFactors)

    def __init__(self):
        r"""Record exact-isotropic-prior and explicit-block-scaling conventions.

        Notes:
            O(1) constructor work; no curvature factors are built or cached.
        """
        super().__init__(
            prior_addition="exact_isotropic", curvature_scaling="explicit_per_block"
        )

    def _check_inputs(self, inputs):
        r"""Inspect scalar and factor metadata without eigendecomposition.

        Args:
            inputs (LaplaceTerms or KFACFactors): Prepared densities and either
                a ready total-precision logdet or explicit factor blocks.

        Returns:
            RequirementReport: Metadata readiness and declared assumptions.

        Raises:
            InputError: If dimensions, factor types/shapes, known scalar
                metadata, dtype/device agreement or conventions are invalid.

        Notes:
            O(B) work for B blocks, O(1) for LaplaceTerms. Tensor values are not
            scanned; symmetry, finiteness and PSD checks happen in estimate.
        """
        scalar_metadata(inputs.log_likelihood, "log_likelihood")
        scalar_metadata(inputs.log_prior, "log_prior")
        if isinstance(inputs, LaplaceTerms):
            positive_int(inputs.dimension, "dimension")
            scalar_metadata(inputs.logdet_precision, "logdet_precision")
        else:
            if not isinstance(inputs.curvature_kind, str) or not inputs.curvature_kind:
                raise InputError(
                    "curvature_kind must name the supplied curvature convention"
                )
            scalar_metadata(inputs.prior_precision, "prior_precision")
            if not isinstance(inputs.prior_precision, torch.Tensor):
                if (
                    not math.isfinite(inputs.prior_precision)
                    or inputs.prior_precision <= 0
                ):
                    raise InputError("prior_precision must be finite and positive")
            if not isinstance(inputs.blocks, (tuple, list)) or not inputs.blocks:
                raise InputError("At least one explicit Kronecker block is required")
            reference = None
            for block in inputs.blocks:
                if not isinstance(block, KroneckerBlock):
                    raise InputError("Each block must be a KroneckerBlock")
                scalar_metadata(block.scale, "block.scale")
                if not isinstance(block.scale, torch.Tensor):
                    if not math.isfinite(block.scale) or block.scale <= 0:
                        raise InputError("Each block scale must be finite and positive")
                for name, factor in (
                    ("activation_factor", block.activation_factor),
                    ("output_factor", block.output_factor),
                ):
                    if (
                        not isinstance(factor, torch.Tensor)
                        or not factor.is_floating_point()
                        or factor.ndim != 2
                        or factor.shape[0] == 0
                        or factor.shape[0] != factor.shape[1]
                    ):
                        raise InputError(
                            f"{name} must be a nonempty square floating Tensor"
                        )
                    if reference is None:
                        reference = factor
                    if (
                        factor.dtype != reference.dtype
                        or factor.device != reference.device
                    ):
                        raise InputError(
                            "All Kronecker factors must have matching dtype/device"
                        )
            if isinstance(inputs.log_likelihood, torch.Tensor):
                if (
                    inputs.log_likelihood.dtype != reference.dtype
                    or inputs.log_likelihood.device != reference.device
                ):
                    raise InputError(
                        "Factors and likelihood must have matching dtype/device"
                    )
        # Symmetry, finiteness and PSD are numerical checks performed only by
        # estimate; preflight does not scan tensors or compute eigendecompositions.
        return super()._check_inputs(inputs)

    def _estimate(self, inputs):
        r"""Assemble the Gaussian integral approximation in natural-log units.

        Args:
            inputs (LaplaceTerms or KFACFactors): Ready terms or total-scaled
                Kronecker factors with a normalised isotropic prior.

        Returns:
            ComplexityResult: Approximate negative log evidence and separately
            named likelihood, prior, half-logdet and Gaussian-normaliser parts.

        Raises:
            InputError: If scalar values, PSD factors, precision or conventions
                cannot define a finite positive-precision calculation.

        Notes:
            LaplaceTerms cost O(1). For factor sizes a_b and g_b, work is
            O(sum_b(a_b^3+g_b^3+a_b*g_b)); per-block workspace is
            O(a_b^2+g_b^2+a_b*g_b), excluding retained autograd graphs. Never
            forms a dense (a_b*g_b)-square precision matrix. Tensor operations
            preserve supported gradients, dtype and device.
        """
        ll = scalar(inputs.log_likelihood, "log_likelihood")
        log_prior = scalar(inputs.log_prior, "log_prior")
        if (
            isinstance(inputs, KFACFactors)
            and inputs.blocks
            and isinstance(inputs.blocks[0].activation_factor, torch.Tensor)
            and not isinstance(inputs.log_likelihood, torch.Tensor)
        ):
            ll = inputs.blocks[0].activation_factor.new_tensor(inputs.log_likelihood)
        if not isinstance(inputs.log_prior, torch.Tensor):
            log_prior = ll.new_tensor(inputs.log_prior)
        if isinstance(inputs, KFACFactors):
            if not inputs.curvature_kind:
                raise InputError(
                    "curvature_kind must name the supplied curvature convention"
                )
            if not math.isfinite(inputs.prior_precision) or inputs.prior_precision <= 0:
                raise InputError("prior_precision must be finite and positive")
            if not inputs.blocks:
                raise InputError("At least one Kronecker block is required")
            logdet = ll.new_zeros(())
            dimension = 0
            for block in inputs.blocks:
                if not math.isfinite(block.scale) or block.scale <= 0:
                    raise InputError("Each block scale must be finite and positive")
                a = _psd_eigenvalues(block.activation_factor, "activation_factor")
                g = _psd_eigenvalues(block.output_factor, "output_factor")
                if (
                    a.device != g.device
                    or a.dtype != g.dtype
                    or a.device != ll.device
                    or a.dtype != ll.dtype
                ):
                    raise InputError(
                        "Factors and likelihood must have matching dtype/device"
                    )
                eigenvalues = (
                    block.scale * g[:, None] * a[None, :] + inputs.prior_precision
                )
                logdet = logdet + eigenvalues.log().sum()
                dimension += a.numel() * g.numel()
        else:
            dimension = positive_int(inputs.dimension, "dimension")
            logdet = scalar(inputs.logdet_precision, "logdet_precision")
            if not isinstance(inputs.logdet_precision, torch.Tensor):
                logdet = ll.new_tensor(inputs.logdet_precision)
        gaussian_normalizer = ll.new_tensor(-0.5 * dimension * math.log(2 * math.pi))
        value = -ll - log_prior + gaussian_normalizer + 0.5 * logdet
        return ComplexityResult(
            method="KFACLaplace",
            kind="negative_log_evidence_approximation",
            quantities={
                "negative_log_evidence": Quantity(value, "nats"),
                "negative_log_likelihood": Quantity(-ll, "nats"),
                "negative_log_prior": Quantity(-log_prior, "nats"),
                "half_logdet_precision": Quantity(0.5 * logdet, "nats"),
                "gaussian_normalizer": Quantity(gaussian_normalizer, "nats"),
            },
            primary_name="negative_log_evidence",
            config={
                "dimension": dimension,
                "curvature_kind": getattr(
                    inputs, "curvature_kind", "supplied_total_precision"
                ),
                **(
                    {
                        "prior_precision": inputs.prior_precision,
                        "block_scales": tuple(block.scale for block in inputs.blocks),
                        "block_shapes": tuple(
                            (
                                tuple(block.activation_factor.shape),
                                tuple(block.output_factor.shape),
                            )
                            for block in inputs.blocks
                        ),
                    }
                    if isinstance(inputs, KFACFactors)
                    else {}
                ),
            },
            assumptions=(
                "Reference point must be MAP for the declared normalised prior.",
                "Selected parameter dimension and total-data curvature must agree.",
                "KFAC/block independence is an approximation except in special Gaussian cases.",
            ),
            provenance=inputs.provenance,
        )
