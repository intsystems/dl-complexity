"""Bayesian objectives expose their assumptions independently of coding APIs."""

from typing import Generic, TypeVar

from dl_complexity.base import ComplexityEstimator

InputsT = TypeVar("InputsT")


class BayesianEstimator(ComplexityEstimator[InputsT], Generic[InputsT]):
    r"""Base interface for variational free energies and evidence approximations.

    Notes:
        Family membership does not identify a single estimand or make the
        result a decodable bitstream. Typed inputs preserve distinctions among
        ordinary/tempered expectations, variational KL and posterior curvature.
        This empty semantic base adds no numeric work to ComplexityEstimator.
    """
