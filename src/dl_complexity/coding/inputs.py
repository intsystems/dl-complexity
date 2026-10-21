"""Immutable descriptions of code costs and causal predictive events."""

from dataclasses import dataclass

import torch

from ..results import Provenance


@dataclass(frozen=True)
class TwoPartInputs:
    r"""Model bits plus data nats for the *decoded*, not original, predictor.

    ``discrete`` requires a nonnegative data code. For a continuous likelihood,
    provide the resolution adjustment explicitly and select ``quantized``.
    The adjustment is the negative total log-cell-volume, in nats. Density times
    cell volume is an approximation, not an exact integration of bin probability.

    Args:
        model_bits (Tensor or float): Complete nonnegative binary model cost.
        data_nll (Tensor or float): Total natural-log loss of the decoded predictor.
        n (int): Number of observations, not batches.
        provenance (Provenance, optional): User-declared state/data identities.
        data_code_convention (str): ``discrete`` or ``quantized`` density approximation.
        resolution_adjustment_nats (Tensor or float): Negative total log-cell-volume.

    Notes:
        Construction stores references in O(1); it does not encode or evaluate a model.
        Frozen records do not make the referenced tensors immutable.
    """

    model_bits: torch.Tensor | float
    data_nll: torch.Tensor | float
    n: int
    provenance: Provenance | None = None
    data_code_convention: str = "discrete"
    resolution_adjustment_nats: torch.Tensor | float = 0.0


@dataclass(frozen=True)
class PrequentialEntry:
    r"""One scored interval [start, stop), before its labels enter training.

    ``trained_until`` and ``calibrated_until`` are exclusive prefix endpoints.
    ``replay_indices`` describes the update *after* scoring this interval, so
    indices in the just-revealed interval are allowed, future indices are not.

    Args:
        start (int): Inclusive first observation scored by this predictor.
        stop (int): Exclusive last observation, with start < stop.
        nll (Tensor or float): Total categorical code cost in nats for this interval.
        trained_until (int): Exclusive endpoint of labels used before prediction.
        calibrated_until (int): Exclusive calibration prefix endpoint.
        replay_indices (tuple[int]): Indices used only after this block was scored.

    Notes:
        O(1) record creation; replay index storage is O(R) when the caller supplies R
        indices. The record itself does not prove that an external callback was causal.
    """

    start: int
    stop: int
    nll: torch.Tensor | float
    trained_until: int
    calibrated_until: int = 0
    replay_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class PrequentialLedger:
    r"""Initial code and exactly-once scores for every remaining observation.

    Args:
        initial_nll (Tensor or float): Initial prefix cost in nats.
        initial_count (int): Number of observations encoded by that initial cost.
        entries (tuple[PrequentialEntry]): Ordered, gap-free prediction intervals.
        n (int): Total sequence length.
        protocol (str): ``blockwise``, ``online`` or ``replay``.
        provenance (Provenance, optional): User-declared identities.

    Notes:
        O(1) wrapper creation; retained storage is O(B + R_total) for B intervals
        and all recorded replay indices, in addition to the caller-owned sequence.
    """

    initial_nll: torch.Tensor | float
    initial_count: int
    entries: tuple[PrequentialEntry, ...]
    n: int
    protocol: str
    provenance: Provenance | None = None
