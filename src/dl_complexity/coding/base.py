"""Code-length estimators share a contract, not an obligatory training step."""

from typing import Generic, TypeVar

from ..base import ComplexityEstimator

TInputs = TypeVar("TInputs")


class CodeLengthEstimator(ComplexityEstimator[TInputs], Generic[TInputs]):
    """Base for explicitly defined model/data codes and predictive protocols."""
