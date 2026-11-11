"""An explicit selection of weight tensors for structural diagnostics."""

from dataclasses import dataclass
from typing import Mapping

import torch

from ..results import Provenance


@dataclass(frozen=True)
class WeightInputs:
    r"""Declare named tensors whose structural/norm statistics should be measured.

    Args:
        weights (Mapping[str, torch.Tensor]): Floating weight tensors; aliases allowed.
        provenance (Provenance, optional): Declared model identity.

    Notes:
        The record retains references, not an independent checkpoint snapshot.
    """

    weights: Mapping[str, torch.Tensor]
    provenance: Provenance | None = None
