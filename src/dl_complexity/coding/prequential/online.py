"""Predict-before-update online prequential code."""

from .base import PrequentialEstimator


class OnlinePrequential(PrequentialEstimator):
    r"""Evaluate a online ledger for one incrementally updated predictor.

    Args:
        unit (str): Inherited output unit, ``nats`` or ``bits``.

    Notes:
        ``prepare_online`` constructs the ledger through explicit callbacks.
        The inherited estimate validates B intervals and R recorded replay indices,
        then sums B scalar costs in O(B + R) time and O(B) temporary storage.
        It never reconstructs prefix predictions from final checkpoint weights.
    """

    protocol = "online"
