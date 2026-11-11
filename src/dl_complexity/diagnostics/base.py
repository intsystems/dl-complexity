"""Diagnostics share the invocation protocol, not a codelength interpretation."""

from ..base import ComplexityEstimator


class DiagnosticEstimator(ComplexityEstimator):
    r"""Share check/estimate for diagnostics without pretending they are code lengths."""
