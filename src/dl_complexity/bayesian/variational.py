"""Negative ELBO = E_q[-log p(D|theta)] + KL(q||prior).

Graves (2011), Practical Variational Inference for Neural Networks, equations
2--6. No data averaging, minibatch scaling or weight decay is inserted here.
"""

from torch import Tensor

from dl_complexity._validation import positive_int, scalar
from dl_complexity.bayesian._checks import scalar_metadata
from dl_complexity.bayesian.base import BayesianEstimator
from dl_complexity.bayesian.inputs import VariationalTerms
from dl_complexity.errors import InputError
from dl_complexity.results import ComplexityResult, Quantity


class VariationalCode(BayesianEstimator[VariationalTerms]):
    r"""Return total variational free energy, not a physical coded bitstream.

    Formula::

        negative_elbo = expected_nll + kl_divergence

    Notes:
        Both terms are total quantities in nats. Dataset averaging, minibatch
        multiplication, KL annealing, weight decay and fitting are not inserted.
        Code interpretation needs additional coding assumptions. The inherited
        constructor stores only configuration, with no data/model execution.
    """

    input_type = VariationalTerms

    def _check_inputs(self, inputs):
        r"""Inspect scalar metadata, observation count and KL-estimator label.

        Args:
            inputs (VariationalTerms): Ready scalar objective terms.

        Returns:
            RequirementReport: Metadata readiness with inherited assumptions.

        Raises:
            InputError: If term shape/type, n or KL-estimator metadata is invalid.

        Notes:
            O(1) time/storage; no samples, tensor value scans or fitting.
        """
        scalar_metadata(inputs.expected_nll, "expected_nll")
        scalar_metadata(inputs.kl_divergence, "kl_divergence")
        if inputs.n is not None:
            positive_int(inputs.n, "n")
        if inputs.kl_estimator not in {"analytic", "monte_carlo", "supplied"}:
            raise InputError("kl_estimator must be analytic, monte_carlo or supplied")
        return super()._check_inputs(inputs)

    def _estimate(self, inputs):
        r"""Add total expected NLL and KL while preserving tensor conventions.

        Args:
            inputs (VariationalTerms): Finite scalar terms in natural logs.

        Returns:
            ComplexityResult: Negative ELBO, expected NLL and KL in nats.

        Raises:
            InputError: If terms/counts are invalid or an analytic KL is negative.

        Notes:
            O(1) work/storage, excluding already existing input graphs. Python
            constants adopt a supplied Tensor's dtype/device. Supplied Tensor
            terms must be mutually compatible; no silent device migration is
            performed. Finite-sample MC KL can be negative and is not clipped.
        """
        nll = scalar(inputs.expected_nll, "expected_nll")
        kl = scalar(inputs.kl_divergence, "kl_divergence")
        # Treat Python constants as constants of the supplied tensor, not as an
        # implicit float64/device migration of a float32/CUDA objective.
        if isinstance(inputs.expected_nll, Tensor) and not isinstance(
            inputs.kl_divergence, Tensor
        ):
            kl = nll.new_tensor(inputs.kl_divergence)
        elif isinstance(inputs.kl_divergence, Tensor) and not isinstance(
            inputs.expected_nll, Tensor
        ):
            nll = kl.new_tensor(inputs.expected_nll)
        if inputs.n is not None:
            positive_int(inputs.n, "n")
        if inputs.kl_estimator not in {"analytic", "monte_carlo", "supplied"}:
            raise InputError("kl_estimator must be analytic, monte_carlo or supplied")
        if inputs.kl_estimator == "analytic" and bool(kl < 0):
            raise InputError("An analytic KL divergence cannot be negative")
        return ComplexityResult(
            method="VariationalCode",
            kind="variational_free_energy",
            quantities={
                "negative_elbo": Quantity(nll + kl, "nats"),
                "expected_nll": Quantity(nll, "nats"),
                "kl_divergence": Quantity(kl, "nats"),
            },
            primary_name="negative_elbo",
            config={"n": inputs.n, "kl_estimator": inputs.kl_estimator},
            assumptions=(
                "Normalised likelihood, prior and q; total-data NLL, no hidden minibatch scaling.",
                "Code interpretation needs additional coding assumptions; this is not a physical codec.",
            ),
            provenance=inputs.provenance,
        )
