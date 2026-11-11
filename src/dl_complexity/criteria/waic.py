"""WAIC2 using sample variance of pointwise posterior log-likelihood.

Vehtari, Gelman, Gabry (2017), arXiv:1507.04544, equations 11--13.
The reported primary value is the deviance-scale -2 (lppd - p_waic).
"""

import math

import torch

from dl_complexity._validation import as_tensor
from dl_complexity.criteria.base import InformationCriterion
from dl_complexity.criteria.inputs import PosteriorLogLikelihood
from dl_complexity.errors import InputError
from dl_complexity.results import ComplexityResult, Quantity


class WAIC(InformationCriterion[PosteriorLogLikelihood]):
    r"""Compute deviance-scale WAIC2 with sample variance (ddof=1).

    Formula::

        lppd = sum_i log(mean_s(exp(log_likelihood[s, i])))
        p_waic = sum_i Var_s(log_likelihood[s, i], correction=1)
        elpd_waic = lppd - p_waic
        waic = -2 * elpd_waic

    Notes:
        Inputs are a [S, N] matrix with S >= 2 ordinary posterior draws. The
        sample correction and observation factorisation are explicit; a single
        deterministic checkpoint cannot supply posterior uncertainty.
        This reducer performs O(S*N) work and retains tensor autograd. It does
        not implement posterior sampling or Monte Carlo error estimation.
    """

    input_type = PosteriorLogLikelihood

    def __init__(self):
        r"""Fix the WAIC2, sample-variance and deviance-scale conventions.

        Notes:
            Stores a small configuration in O(1), without data or posterior work.
        """
        super().__init__(variant="WAIC2", variance_ddof=1, scale="deviance")

    def _check_inputs(self, inputs):
        r"""Inspect posterior matrix rank, dimensions and dtype only.

        Args:
            inputs (PosteriorLogLikelihood): Declared [S, N] likelihood matrix.

        Returns:
            RequirementReport: Inherited readiness/provenance assumptions.

        Raises:
            InputError: If the matrix is not floating with S >= 2 and N >= 1.

        Notes:
            O(1) metadata inspection. Finiteness scanning is deferred to
            ``_estimate``; no sampler or model forward is invoked here.
        """
        ll = inputs.log_likelihood
        if (
            not isinstance(ll, torch.Tensor)
            or ll.ndim != 2
            or ll.shape[0] < 2
            or ll.shape[1] < 1
        ):
            raise InputError(
                "WAIC requires a log_likelihood Tensor shaped [S >= 2, n >= 1]"
            )
        if not ll.is_floating_point():
            raise InputError("WAIC log_likelihood must be floating point")
        return super()._check_inputs(inputs)

    def _estimate(self, inputs):
        r"""Reduce the posterior matrix to lppd, optimism correction and WAIC2.

        Args:
            inputs (PosteriorLogLikelihood): Finite [S, N] natural-log matrix.

        Returns:
            ComplexityResult: Native WAIC, lppd/elpd in nats, and dimensionless
            p_waic effective-parameter correction under the chosen convention.

        Raises:
            InputError: If shape, dtype or numerical values are invalid.

        Notes:
            logsumexp avoids overflow/underflow from direct exponentiation.
            Finiteness, logsumexp and sample variance take O(S*N) work. Input
            and validation/autograd intermediates can require O(S*N) memory;
            the returned quantities are scalars. No posterior is fitted.
        """
        ll = as_tensor(inputs.log_likelihood)
        if ll.ndim != 2 or ll.shape[0] < 2 or ll.shape[1] < 1:
            raise InputError("WAIC requires log_likelihood shaped [S >= 2, n >= 1]")
        if not ll.is_floating_point() or not bool(torch.isfinite(ll).all()):
            raise InputError("WAIC log_likelihood must contain finite floating values")
        lppd = (torch.logsumexp(ll, dim=0) - math.log(ll.shape[0])).sum()
        p_waic = ll.var(dim=0, correction=1).sum()
        elpd = lppd - p_waic
        return ComplexityResult(
            method="WAIC",
            kind="information_criterion",
            quantities={
                "waic": Quantity(-2 * elpd, "native_ic"),
                "lppd": Quantity(lppd, "nats"),
                "p_waic": Quantity(p_waic, "dimensionless"),
                "elpd_waic": Quantity(elpd, "nats"),
            },
            primary_name="waic",
            config={
                "variant": "WAIC2",
                "variance_ddof": 1,
                "posterior_draws": ll.shape[0],
                "n": ll.shape[1],
            },
            assumptions=(
                "Ordinary posterior draws and pointwise observation factorisation supplied by caller.",
                "Monte Carlo error is not estimated by this reducer.",
            ),
            provenance=inputs.provenance,
        )
