"""Base contract for generalization-risk bounds, not code-length estimators."""

from typing import Generic, TypeVar

from ..base import ComplexityEstimator

TInputs = TypeVar("TInputs")


class GeneralizationBoundEstimator(ComplexityEstimator[TInputs], Generic[TInputs]):
    """A bound must identify its theorem, predictor and confidence convention."""
