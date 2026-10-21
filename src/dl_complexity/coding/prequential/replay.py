"""Prequential code with revealed-prefix replay, not duplicate data coding."""

from .base import PrequentialEstimator


class ReplayPrequential(PrequentialEstimator):
    r"""Evaluate a replay ledger for one predictor updated with revealed-prefix replay.

    Args:
        unit (str): Inherited output unit, ``nats`` or ``bits``.

    Notes:
        ``prepare_replay`` constructs the ledger through explicit callbacks.
        The inherited estimate validates B intervals and R recorded replay indices,
        then sums B scalar costs in O(B + R) time and O(B) temporary storage.
        It never reconstructs prefix predictions from final checkpoint weights.
    """

    protocol = "replay"
