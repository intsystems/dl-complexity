"""Hannan-Quinn criterion: -2 log L + 2 k log log n.

Hannan and Quinn (1979), doi:10.1111/j.2517-6161.1979.tb01072.x.
The native formula is retained even when its small-n penalty is negative.
"""

import math

from dl_complexity.criteria._fitted import FittedCriterion
from dl_complexity.errors import InputError
from dl_complexity.results import ComplexityResult, Quantity


class HQIC(FittedCriterion):
    r"""Compute the native information criterion ``HQIC``.

    Formula::

        -2 * log_likelihood + 2 * k * log(log(n))

    Args:
        parameter_count (str or int): Context-route parameter selection;
            inherited from FittedCriterion. Prepared inputs use their own k.

    Notes:
        Supply total natural log-likelihood, not average minibatch loss.
        For n=2 the native penalty is negative and is not clipped.
        The result uses ``native_ic``, with separate deviance and penalty.
        Constructor and prepared-input evaluation take O(1) work/storage;
        a context route additionally evaluates model likelihood.

    Examples:
        >>> from dl_complexity.criteria import FitStatistics
        >>> result = HQIC().estimate(inputs=FitStatistics(-100.0, 3, 100))
        >>> result.primary.unit
        'native_ic'
    """

    needs_n = True

    def _check_inputs(self, inputs):
        r"""Extend the fitted-statistics preflight with the log-log domain.

        Args:
            inputs (FitStatistics): Prepared likelihood/count metadata.

        Returns:
            RequirementReport: The inherited metadata-only readiness report.

        Raises:
            InputError: If a supplied n is at most one, or inherited checks fail.

        Notes:
            O(1); this checks n without reading likelihood values or running
            the model. n=2 remains valid despite a negative native penalty.
        """
        report = super()._check_inputs(inputs)
        if inputs.n is not None and inputs.n <= 1:
            raise InputError("HQIC requires n > 1 because log(log(n)) is undefined")
        return report

    def _estimate(self, inputs):
        r"""Combine native deviance with the HQIC penalty.

        Args:
            inputs (FitStatistics): Prepared scalar total log-likelihood and
                fitted-parameter metadata; n is required when ``needs_n``.

        Returns:
            ComplexityResult: Native criterion, deviance and penalty scalars;
            provenance and k/n are retained separately from assumptions.

        Raises:
            InputError: If numeric statistics do not satisfy the formula's domain.

        Notes:
            Computes ``-2 * log_likelihood + 2 * k * log(log(n))`` in O(1) time/memory.
            Tensor operations preserve the log-likelihood gradient; counts
            and Python log penalties are not learnable model parameters.
        """
        ll, k, n = self._statistics(inputs)
        if n <= 1:
            raise InputError("HQIC requires n > 1 because log(log(n)) is undefined")
        deviance = -2 * ll
        penalty = ll.new_tensor(2 * k * math.log(math.log(n)))
        return ComplexityResult(
            method="HQIC",
            kind="information_criterion",
            quantities={
                "hqic": Quantity(deviance + penalty, "native_ic"),
                "deviance": Quantity(deviance, "native_ic"),
                "penalty": Quantity(penalty, "native_ic"),
            },
            primary_name="hqic",
            config={"k": k, "n": n},
            assumptions=(
                "Hannan-Quinn native penalty; applicable asymptotic conditions must be checked separately.",
            ),
            provenance=inputs.provenance,
        )
