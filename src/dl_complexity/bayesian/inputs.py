"""Explicit Bayesian numerical artifacts; none imply hidden fitting/sampling."""

from dataclasses import dataclass

from torch import Tensor

from dl_complexity.results import Provenance


@dataclass(frozen=True)
class TemperedLikelihood:
    r"""Store total likelihood evaluations from a declared tempered posterior.

    Args:
        log_likelihood (Tensor): Floating vector [S] of total natural
            log-likelihood values, one per tempered-posterior draw.
        n (int): Number of observations, greater than one.
        beta (float): Inverse temperature; WBIC requires 1/log(n).
        provenance (Provenance, optional): Declared artifact identities.

    Notes:
        The target is proportional to prior * likelihood**beta. Ordinary
        posterior draws cannot silently replace this target. Construction is
        O(1) reference storage; the caller-owned vector uses O(S) memory.
        The record does not run a sampler or establish mixing/convergence.
    """

    log_likelihood: Tensor
    n: int
    beta: float
    provenance: Provenance | None = None


@dataclass(frozen=True)
class VariationalTerms:
    r"""Store total negative-ELBO terms without implicit dataset rescaling.

    Args:
        expected_nll (float or Tensor): Scalar E_q[-log p(D|theta)], total over D.
        kl_divergence (float or Tensor): Scalar KL(q||prior) estimate, in nats.
        n (int, optional): Observation count, recorded but not used to scale KL.
        kl_estimator (str): ``"analytic"``, ``"monte_carlo"`` or ``"supplied"``.
        provenance (Provenance, optional): User-declared artifact identities.

    Notes:
        Construction stores references in O(1). Analytic KL must be nonnegative;
        a finite-sample Monte Carlo KL estimate may be negative. Differentiable
        tensors retain their graph; this record does not train q or the model.
    """

    expected_nll: Tensor | float
    kl_divergence: Tensor | float
    n: int | None = None
    kl_estimator: str = "analytic"
    provenance: Provenance | None = None


@dataclass(frozen=True)
class LaplaceTerms:
    r"""Store reference-point densities and total posterior precision logdet.

    Args:
        log_likelihood (float or Tensor): Scalar total natural log-likelihood.
        log_prior (float or Tensor): Scalar normalised prior log-density at the
            same reference point, including its normalising constant.
        dimension (int): Positive selected parameter dimension d.
        logdet_precision (float or Tensor): Scalar log determinant of the total
            posterior precision, not an unscaled average-data curvature.
        provenance (Provenance, optional): User-declared artifact identities.

    Notes:
        This O(1) record can bypass a new curvature pass. MAP, valid precision,
        selected subset and agreement of all terms remain caller assumptions.
    """

    log_likelihood: Tensor | float
    log_prior: Tensor | float
    dimension: int
    logdet_precision: Tensor | float
    provenance: Provenance | None = None


@dataclass(frozen=True)
class KroneckerBlock:
    r"""Describe a likelihood precision block as scale * (G kron A).

    Args:
        activation_factor (Tensor): Symmetric PSD floating matrix A, [a, a].
        output_factor (Tensor): Symmetric PSD floating matrix G, [g, g].
        scale (float): Positive multiplier converting the supplied factor
            convention to total-data likelihood precision.

    Notes:
        Construction is O(1), storing references to O(a^2+g^2) input values.
        The implicit block dimension is a*g; the dense block is not formed.
        Prior addition and numerical factor validation occur in KFACLaplace.
    """

    activation_factor: Tensor
    output_factor: Tensor
    scale: float = 1.0


@dataclass(frozen=True)
class KFACFactors:
    r"""Store Kronecker likelihood factors and exact isotropic prior addition.

    Args:
        log_likelihood (float or Tensor): Scalar total natural log-likelihood.
        log_prior (float or Tensor): Scalar normalised prior log-density.
        blocks (tuple of KroneckerBlock): Nonempty selected block decomposition.
        prior_precision (float): Positive isotropic precision alpha.
        curvature_kind (str): Explicit Hessian/GGN/Fisher convention label.
        provenance (Provenance, optional): User-declared artifact identities.

    Formula::

        block_eigenvalues[j, i] = scale * eig(G)[j] * eig(A)[i] + alpha

    Notes:
        alpha*I is added to each implicit block exactly. This is NOT
        (A+aI) kron (G+bI), which introduces different cross terms. Construction
        stores references in O(1); no general-network factor collector is run.
    """

    log_likelihood: Tensor | float
    log_prior: Tensor | float
    blocks: tuple[KroneckerBlock, ...]
    prior_precision: float
    curvature_kind: str
    provenance: Provenance | None = None
