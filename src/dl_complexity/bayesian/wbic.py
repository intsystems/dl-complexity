"""WBIC = E_beta[-log L], beta = 1/log(n), with no extra beta or 2 factor.

Watanabe (2013), JMLR 14:867--897, equations 1--5:
https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf
"""

import math

import torch

from dl_complexity._validation import as_tensor, positive_int
from dl_complexity.bayesian.base import BayesianEstimator
from dl_complexity.bayesian.inputs import TemperedLikelihood
from dl_complexity.errors import InputError
from dl_complexity.results import ComplexityResult, Quantity


class WBIC(BayesianEstimator[TemperedLikelihood]):
    r"""Average total NLL under the explicitly declared WBIC temperature.

    Formula::

        beta = 1 / log(n)
        wbic = -mean_s(total_log_likelihood[s])

    Notes:
        No extra beta or deviance factor 2 multiplies the mean. Inputs must
        target prior * likelihood**beta; neither ordinary posterior draws nor
        repeated predictions from a deterministic checkpoint provide this.
        Sampling/mixing and the asymptotic evidence interpretation are separate
        assumptions, not established by this reducer.
    """

    input_type = TemperedLikelihood

    def _check_inputs(self, inputs):
        r"""Check n, beta and vector metadata without reading likelihood values.

        Args:
            inputs (TemperedLikelihood): Total likelihood vector [S] and n/beta.

        Returns:
            RequirementReport: Readiness with declared provenance assumptions.

        Raises:
            InputError: If n <= 1, beta does not equal 1/log(n) within the fixed
                tolerance, or likelihood metadata is not floating [S >= 1].

        Notes:
            O(1) time/storage; does not sample or scan the vector contents.
        """
        positive_int(inputs.n, "n")
        if inputs.n <= 1:
            raise InputError("WBIC requires n > 1")
        expected_beta = 1 / math.log(inputs.n)
        if not math.isfinite(inputs.beta) or not math.isclose(
            inputs.beta, expected_beta, rel_tol=1e-6, abs_tol=1e-12
        ):
            raise InputError("WBIC requires tempered draws with beta=1/log(n)")
        ll = inputs.log_likelihood
        if (
            not isinstance(ll, torch.Tensor)
            or ll.ndim != 1
            or ll.numel() < 1
            or not ll.is_floating_point()
        ):
            raise InputError(
                "WBIC requires a floating log_likelihood Tensor shaped [S >= 1]"
            )
        return super()._check_inputs(inputs)

    def _estimate(self, inputs):
        r"""Compute the sample mean of total tempered-posterior negative log likelihood.

        Args:
            inputs (TemperedLikelihood): Finite floating vector [S] in nats.

        Returns:
            ComplexityResult: WBIC in nats, plus n/beta/draw-count configuration.

        Raises:
            InputError: If counts, temperature, vector shape or values are invalid.

        Notes:
            O(S) work for finite-value validation and reduction; O(S) input and
            possible validation/autograd storage, with a scalar output. Existing
            tensor dtype/device/autograd are preserved. No sampler or variance
            estimate is created.
        """
        n = positive_int(inputs.n, "n")
        if n <= 1:
            raise InputError("WBIC requires n > 1")
        expected_beta = 1 / math.log(n)
        if not math.isfinite(inputs.beta) or not math.isclose(
            inputs.beta, expected_beta, rel_tol=1e-6, abs_tol=1e-12
        ):
            raise InputError("WBIC requires tempered draws with beta=1/log(n)")
        ll = as_tensor(inputs.log_likelihood)
        if (
            ll.ndim != 1
            or ll.numel() == 0
            or not ll.is_floating_point()
            or not bool(torch.isfinite(ll).all())
        ):
            raise InputError(
                "WBIC requires finite total log_likelihood shaped [S >= 1]"
            )
        return ComplexityResult(
            method="WBIC",
            kind="negative_log_evidence_approximation",
            quantities={"wbic": Quantity(-ll.mean(), "nats")},
            primary_name="wbic",
            config={"beta": inputs.beta, "n": n, "tempered_draws": ll.numel()},
            assumptions=(
                "Draws must target prior * likelihood**beta; convergence is not certified.",
                "WBIC is asymptotic and is not exact finite-sample evidence.",
            ),
            provenance=inputs.provenance,
        )
