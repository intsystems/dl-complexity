"""Explicit batch source and observation identity, without hidden downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from ._validation import positive_int
from .errors import InputError


@dataclass(frozen=True)
class DataSource:
    r"""Describe observation batches without silently downloading or materializing them.

    Args:
        batches (Iterable or Callable): Batches (inputs, targets), or a fresh factory.
        n_observations (int, optional): Declared count, validated after consumption.
        data_id (str, optional): Declared dataset/preprocessing identity.

    Notes:
        Targets' first axis indexes observations; likelihood sums event axes.
        A one-shot iterator is permitted for one explicit evaluation, not shared
        consumption by several suite readers. Construction does not iterate.
    """

    batches: Iterable | Callable[[], Iterable]
    n_observations: int | None = None
    data_id: str | None = None

    def __post_init__(self):
        r"""Validate a declared positive observation count without touching batches."""
        if self.n_observations is not None:
            positive_int(self.n_observations, "n_observations")

    @classmethod
    def from_tensors(cls, inputs: torch.Tensor, targets: torch.Tensor, *, data_id=None):
        r"""Wrap tensors as one reusable full batch without copying their values.

        Args:
            inputs (torch.Tensor): Shape [N, ...], sharing first axis with targets.
            targets (torch.Tensor): Shape [N, ...] with N positive.
            data_id (str, optional): Dataset identity.

        Returns:
            DataSource: Single-batch iterable retaining references to the input tensors.

        Raises:
            InputError: Missing observation axis or inconsistent lengths.

        Notes:
            Metadata-only O(1) wrapping, but the caller already owns the full dataset.
        """
        if inputs.ndim == 0 or targets.ndim == 0 or len(inputs) != len(targets):
            raise InputError(
                "Inputs and targets must share a nonempty observation axis"
            )
        return cls(((inputs, targets),), len(targets), data_id)

    def __iter__(self):
        r"""Create a batch iterator, calling the factory only at explicit iteration time.

        Returns:
            Iterator: Underlying iterator; arbitrary factory costs belong to the caller.
        """
        return iter(self.batches() if callable(self.batches) else self.batches)

    @property
    def one_shot(self) -> bool:
        # Structural check only: never invoke __iter__ or a batch factory here.
        r"""Detect an existing iterator by __next__ without invoking a factory or __iter__.

        Returns:
            bool: Structural reuse warning, not a proof that a custom iterable is repeatable.
        """
        return not callable(self.batches) and hasattr(self.batches, "__next__")
