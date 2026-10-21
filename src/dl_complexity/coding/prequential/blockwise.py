"""Chunk-incremental/from-scratch prequential code."""

from .base import PrequentialEstimator


class BlockwisePrequential(PrequentialEstimator):
    r"""Evaluate a blockwise ledger for independently retrained prefix predictors.

    Args:
        unit (str): Inherited output unit, ``nats`` or ``bits``.

    Notes:
        ``prepare_blockwise`` constructs the ledger through explicit callbacks.
        The inherited estimate validates B intervals and R recorded replay indices,
        then sums B scalar costs in O(B + R) time and O(B) temporary storage.
        It never reconstructs prefix predictions from final checkpoint weights.
    """

    protocol = "blockwise"
