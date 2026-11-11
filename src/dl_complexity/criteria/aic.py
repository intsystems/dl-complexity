"""Akaike information criterion: AIC = -2 log L(theta_hat) + 2 k.

See Akaike (1974), doi:10.1109/TAC.1974.1100705. Native deviance scale;
regularity and the fitted-state interpretation are assumptions, not inferred.
"""

from dl_complexity.criteria._fitted import FittedCriterion
from dl_complexity.results import ComplexityResult, Quantity


class AIC(FittedCriterion):
    r"""Compute the native information criterion ``AIC``.

    Formula::

        -2 * log_likelihood + 2 * k

    Args:
        parameter_count (str or int): Context-route parameter selection;
            inherited from FittedCriterion. Prepared inputs use their own k.

    Notes:
        Supply total natural log-likelihood, not average minibatch loss.
        The observation count is optional for the total score.
        The result uses ``native_ic``, with separate deviance and penalty.
        Constructor and prepared-input evaluation take O(1) work/storage;
        a context route additionally evaluates model likelihood.

    Examples:
        >>> from dl_complexity.criteria import FitStatistics
        >>> result = AIC().estimate(inputs=FitStatistics(-100.0, 3, 100))
        >>> result.primary.unit
        'native_ic'
    """

    def _estimate(self, inputs):
        r"""Combine native deviance with the AIC penalty.

        Args:
            inputs (FitStatistics): Prepared scalar total log-likelihood and
                fitted-parameter metadata; n is required when ``needs_n``.

        Returns:
            ComplexityResult: Native criterion, deviance and penalty scalars;
            provenance and k/n are retained separately from assumptions.

        Raises:
            InputError: If numeric statistics do not satisfy the formula's domain.

        Notes:
            Computes ``-2 * log_likelihood + 2 * k`` in O(1) time/memory.
            Tensor operations preserve the log-likelihood gradient; counts
            and Python log penalties are not learnable model parameters.
        """
        ll, k, n = self._statistics(inputs)
        deviance = -2 * ll
        penalty = ll.new_tensor(2 * k)
        return ComplexityResult(
            method="AIC",
            kind="information_criterion",
            quantities={
                "aic": Quantity(deviance + penalty, "native_ic"),
                "deviance": Quantity(deviance, "native_ic"),
                "penalty": Quantity(penalty, "native_ic"),
            },
            primary_name="aic",
            config={"k": k, "n": n},
            assumptions=("Regular fitted-likelihood asymptotics; MLE not certified.",),
            provenance=inputs.provenance,
        )
