"""Schwarz BIC = -2 log L(theta_hat) + k log n, native scale.

Schwarz (1978), doi:10.1214/aos/1176344136. Half-BIC is only an asymptotic
negative-log-evidence proxy under its assumptions, not a universal MDL code.
"""

import math

from dl_complexity.criteria._fitted import FittedCriterion
from dl_complexity.results import ComplexityResult, Quantity


class BIC(FittedCriterion):
    r"""Compute the native information criterion ``BIC``.

    Formula::

        -2 * log_likelihood + k * log(n)

    Args:
        parameter_count (str or int): Context-route parameter selection;
            inherited from FittedCriterion. Prepared inputs use their own k.

    Notes:
        Supply total natural log-likelihood, not average minibatch loss.
        Half-BIC is not returned as an exact code or exact evidence.
        The result uses ``native_ic``, with separate deviance and penalty.
        Constructor and prepared-input evaluation take O(1) work/storage;
        a context route additionally evaluates model likelihood.

    Examples:
        >>> from dl_complexity.criteria import FitStatistics
        >>> result = BIC().estimate(inputs=FitStatistics(-100.0, 3, 100))
        >>> result.primary.unit
        'native_ic'
    """

    needs_n = True

    def _estimate(self, inputs):
        r"""Combine native deviance with the BIC penalty.

        Args:
            inputs (FitStatistics): Prepared scalar total log-likelihood and
                fitted-parameter metadata; n is required when ``needs_n``.

        Returns:
            ComplexityResult: Native criterion, deviance and penalty scalars;
            provenance and k/n are retained separately from assumptions.

        Raises:
            InputError: If numeric statistics do not satisfy the formula's domain.

        Notes:
            Computes ``-2 * log_likelihood + k * log(n)`` in O(1) time/memory.
            Tensor operations preserve the log-likelihood gradient; counts
            and Python log penalties are not learnable model parameters.
        """
        ll, k, n = self._statistics(inputs)
        deviance = -2 * ll
        penalty = ll.new_tensor(k * math.log(n))
        return ComplexityResult(
            method="BIC",
            kind="information_criterion",
            quantities={
                "bic": Quantity(deviance + penalty, "native_ic"),
                "deviance": Quantity(deviance, "native_ic"),
                "penalty": Quantity(penalty, "native_ic"),
            },
            primary_name="bic",
            config={"k": k, "n": n},
            assumptions=(
                "Regular finite-dimensional MLE asymptotics; not certified for arbitrary neural weights.",
            ),
            provenance=inputs.provenance,
        )
