"""Theorem-specific generalization bounds with explicit assumptions."""

from .base import GeneralizationBoundEstimator
from .compression import (
    AroraCompressionBound,
    arora_inputs_from_logits,
    prepare_arora_compression,
    prepare_arora_projection,
)
from .inputs import AroraCompressionInputs

__all__ = [
    "GeneralizationBoundEstimator",
    "AroraCompressionBound",
    "AroraCompressionInputs",
    "arora_inputs_from_logits",
    "prepare_arora_compression",
    "prepare_arora_projection",
]
