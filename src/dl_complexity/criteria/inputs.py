"""Sufficient likelihood statistics; these records do not certify a fitted MLE."""

from dataclasses import dataclass

from torch import Tensor

from dl_complexity.results import Provenance


@dataclass(frozen=True)
class FitStatistics:
    r"""Store sufficient fitted-likelihood numbers without re-evaluating a model.

    Args:
        log_likelihood (float or Tensor): Scalar sum of natural log-likelihood
            over observations. No regularizer or optimizer weight decay is added.
        k (int): Nonnegative statistical parameter dimension. It is not inferred
            as an effective dimension for a singular neural network.
        n (int, optional): Number of observations, not minibatches. May be omitted
            only for unnormalised AIC.
        provenance (Provenance, optional): User-declared state/data/likelihood IDs.

    Notes:
        Construction stores references in O(1); numeric validation belongs to
        the estimator. A frozen dataclass does not make a referenced Tensor
        immutable and does not certify MLE/regularity assumptions.
    """

    log_likelihood: Tensor | float
    k: int
    n: int | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True)
class PosteriorLogLikelihood:
    r"""Store pointwise log-likelihood under ordinary posterior draws.

    Args:
        log_likelihood (Tensor): Floating matrix [S, N] in natural logs. Each
            column is one observation after summing its event axes; each row
            corresponds to one ordinary posterior draw, not a tempered draw.
        provenance (Provenance, optional): User-declared artifact identities.

    Notes:
        WAIC2 requires S >= 2 and N >= 1. Construction is O(1), holds the matrix
        by reference, and neither samples posterior weights nor checks their
        convergence. The matrix itself requires O(S*N) caller-owned storage.
    """

    log_likelihood: Tensor
    provenance: Provenance | None = None
