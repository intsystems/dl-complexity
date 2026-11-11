"""Typed results preserve mathematical meaning instead of forcing one score."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

import torch

from ._validation import as_tensor, positive_int
from .errors import InputError


@dataclass(frozen=True)
class Provenance:
    r"""Declare optional model, dataset and likelihood identities.

    Args:
        state_id (str, optional): Identity for a fixed model state.
        data_id (str, optional): Identity for observations/preprocessing.
        likelihood_id (str, optional): Identity for likelihood and conventions.

    Notes:
        Matching strings are declarations, not content verification or a cache key.
    """

    state_id: str | None = None
    data_id: str | None = None
    likelihood_id: str | None = None


@dataclass(frozen=True)
class Quantity:
    r"""Keep a tensor together with units and normalization.

    Args:
        value (torch.Tensor): Floating scalar or diagnostic tensor.
        unit (str): For example nats, bits, native_ic or probability.
        normalization (str): Total by default; per_observation is explicit.

    Notes:
        The record is shallowly frozen. Tensor contents/autograd are not detached.
    """

    value: torch.Tensor
    unit: str
    normalization: str = "total"

    def __post_init__(self):
        r"""Preserve floating tensors and require a nonempty unit label.

        Raises:
            InputError: Non-floating tensor or missing units.

        Notes:
            No finite-value or scalar-shape restriction is imposed by this container.
        """
        object.__setattr__(self, "value", as_tensor(self.value))
        if not self.unit:
            raise InputError("A quantity needs explicit units")

    def to(self, unit: str) -> Quantity:
        r"""Convert logarithm base without changing the mathematical estimand.

        Args:
            unit (str): Same unit, or bits/nats when the source is nats/bits.

        Returns:
            Quantity: New scaled tensor; same-unit conversion returns self.

        Raises:
            InputError: Requested conversion would reinterpret IC, risk or a norm.

        Notes:
            bits = nats / ln(2). Scaling takes O(value.numel()) work and preserves
            autograd. This is not an automatic IC-to-codelength conversion.
        """
        if unit == self.unit:
            return self
        if (self.unit, unit) == ("nats", "bits"):
            return replace(self, value=self.value / math.log(2), unit=unit)
        if (self.unit, unit) == ("bits", "nats"):
            return replace(self, value=self.value * math.log(2), unit=unit)
        raise InputError(f"No mathematical conversion from {self.unit} to {unit}")

    def per_observation(self, n: int) -> Quantity:
        r"""Divide a total log/code/IC quantity by a declared observation count.

        Args:
            n (int): Positive number of observations, not minibatches.

        Returns:
            Quantity: Value/n labelled per_observation.

        Raises:
            InputError: Already normalized quantity, incompatible unit or invalid n.

        Notes:
            O(value.numel()) work/storage; probability bounds are not divided by n.
        """
        positive_int(n, "n")
        if self.normalization != "total" or self.unit not in {
            "nats",
            "bits",
            "native_ic",
        }:
            raise InputError(
                "Only total log/code/IC quantities support this normalization"
            )
        return replace(self, value=self.value / n, normalization="per_observation")


@dataclass(frozen=True)
class ComplexityResult:
    r"""Return named mathematical outputs and their interpretation.

    Args:
        method (str): Estimator name.
        kind (str): Estimand, distinct from units or module family.
        quantities (Mapping[str, Quantity]): Primary and component quantities.
        primary_name (str, optional): Primary key; diagnostics can omit it.
        assumptions (tuple[str, ...]): Conditions, approximations and limitations.
        config (Mapping): Constructor settings plus actual calculation metadata.
        provenance (Provenance, optional): Declared input identities.
        elapsed_seconds (float, optional): Host-side interval, not GPU benchmark.

    Notes:
        Mappings/tensors are not recursively frozen. No cross-kind ranking is defined.
    """

    method: str
    kind: str
    quantities: Mapping[str, Quantity]
    primary_name: str | None = None
    assumptions: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    elapsed_seconds: float | None = None

    def __post_init__(self):
        r"""Check result labels, quantity wrappers and optional primary-key membership.

        Notes:
            O(number of named quantities), without evaluating their numeric contents.
        """
        if not self.method or not self.kind or not self.quantities:
            raise InputError(
                "Result needs a method, estimand kind and named quantities"
            )
        if self.primary_name is not None and self.primary_name not in self.quantities:
            raise InputError("Primary quantity must be present")
        if not all(isinstance(q, Quantity) for q in self.quantities.values()):
            raise InputError("Each result entry must be a Quantity")

    @property
    def primary(self) -> Quantity | None:
        r"""Look up the named primary quantity, or return None for multi-output diagnostics.

        Returns:
            Quantity or None: Existing quantity; no tensor is copied.
        """
        return None if self.primary_name is None else self.quantities[self.primary_name]


class RequirementStatus(str, Enum):
    r"""Name computability states without asserting theorem applicability."""

    READY = "READY"
    PREPARATION_REQUIRED = "PREPARATION_REQUIRED"
    MISSING_INPUTS = "MISSING_INPUTS"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class RequirementReport:
    r"""Describe preflight readiness separately from scientific assumptions.

    Args:
        status (RequirementStatus): READY, PREPARATION_REQUIRED, MISSING_INPUTS or UNSUPPORTED.
        missing (tuple[str, ...]): Required input descriptions.
        actions (tuple[str, ...]): Explicit preparation steps for the caller.
        assumptions (tuple[str, ...]): Conditions not proved by readiness.
    """

    status: RequirementStatus
    missing: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        r"""Return whether status equals READY; no numerical computation."""
        return self.status == RequirementStatus.READY
