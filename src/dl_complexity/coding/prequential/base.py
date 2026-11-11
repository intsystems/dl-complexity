"""Common exactly-once ledger evaluator for causal prequential protocols."""

import math

import torch

from ..._validation import positive_int, scalar
from ...errors import InputError
from ...results import ComplexityResult, Quantity, RequirementReport, RequirementStatus
from ..base import CodeLengthEstimator
from ..inputs import PrequentialEntry, PrequentialLedger


class PrequentialEstimator(CodeLengthEstimator[PrequentialLedger]):
    r"""Shared causal-ledger validation and code-length arithmetic.

    Args:
        unit (str): ``nats`` or ``bits``.

    Notes:
        Subclasses only declare a protocol name. Model training lives in explicit
        preparation functions, not the common estimate method.
    """

    input_type = PrequentialLedger
    protocol: str

    def __init__(self, *, unit="nats"):
        r"""Store the protocol and logarithmic output unit in O(1).

        Args:
            unit (str): ``nats`` or ``bits``.

        Raises:
            InputError: If the unit is unsupported.
        """
        if unit not in {"nats", "bits"}:
            raise InputError("A code length uses nats or bits")
        super().__init__(unit=unit, protocol=self.protocol)
        self.unit = unit

    def _check_context(self, context, data):
        r"""Report that an ordinary final model lacks a causal prediction ledger.

        Args:
            context (ModelContext, optional): Model metadata only.
            data (DataSource, optional): Data metadata only; never iterated.

        Returns:
            RequirementReport: Missing inputs or an explicit preparation requirement.

        Notes:
            O(1); no training, data iteration or history inference.
        """
        return RequirementReport(
            (
                RequirementStatus.MISSING_INPUTS
                if context is None or data is None
                else RequirementStatus.PREPARATION_REQUIRED
            ),
            missing=("PrequentialLedger",),
            actions=(
                f"Provide an ordered sequence, factory/update protocol and run explicit prepare_{self.protocol}",
            ),
            assumptions=("A final checkpoint is not a prefix prediction history",),
        )

    def _check_inputs(self, inputs):
        r"""Check ledger causality and return the remaining provenance assumptions.

        Args:
            inputs (PrequentialLedger): Explicitly prepared costs and interval metadata.

        Returns:
            RequirementReport: READY when the metadata are consistent.

        Raises:
            InputError: If intervals, prefixes or replay indices violate the protocol.

        Notes:
            O(B + R_total) metadata work, O(1) auxiliary space.
        """
        self._validate_metadata(inputs)
        return RequirementReport(
            RequirementStatus.READY,
            assumptions=(
                "A ledger's actual prediction/training provenance is user-declared",
            ),
        )

    def _validate_metadata(self, inputs):
        r"""Verify complete, exactly-once intervals and no future-label access.

        Args:
            inputs (PrequentialLedger): Ordered intervals after the initial prefix.

        Raises:
            InputError: If protocol, interval coverage, prefix bounds or indices conflict.

        Notes:
            Walks B entries and R_total replay indices: O(B + R_total) time and O(1)
            extra space. It checks declared metadata, not arbitrary callback closures.
        """
        n = positive_int(inputs.n, "n")
        if inputs.protocol != self.protocol:
            raise InputError(f"Expected {self.protocol} ledger, got {inputs.protocol}")
        if (
            isinstance(inputs.initial_count, bool)
            or not isinstance(inputs.initial_count, int)
            or not 0 <= inputs.initial_count <= n
        ):
            raise InputError("initial_count must lie between zero and n")
        stop = inputs.initial_count
        for entry in inputs.entries:
            if not isinstance(entry, PrequentialEntry):
                raise InputError("Each ledger entry must be a PrequentialEntry")
            if any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (
                    entry.start,
                    entry.stop,
                    entry.trained_until,
                    entry.calibrated_until,
                )
            ):
                raise InputError("Ledger indices must be integers, not booleans")
            if entry.start != stop or not entry.start < entry.stop <= n:
                raise InputError(
                    "Ledger intervals must cover the remaining sequence exactly once"
                )
            if not 0 <= entry.trained_until <= entry.start:
                raise InputError("Predictor was trained using future labels")
            if not 0 <= entry.calibrated_until <= entry.start:
                raise InputError("Predictor was calibrated using future labels")
            if entry.replay_indices and self.protocol != "replay":
                raise InputError("Only replay ledgers contain replay update indices")
            if any(
                isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < entry.stop
                for i in entry.replay_indices
            ):
                raise InputError("Replay accessed an unrevealed observation")
            if isinstance(entry.nll, torch.Tensor) and entry.nll.ndim != 0:
                raise InputError("Each block cost must be scalar")
            stop = entry.stop
        if stop != n:
            raise InputError("Ledger is incomplete")

    def _estimate(self, inputs):
        r"""Sum validated initial and subsequent scalar code costs.

        Args:
            inputs (PrequentialLedger): Nonnegative categorical NLL terms in nats.

        Returns:
            ComplexityResult: Total, initial, predictive and per-observation costs.

        Raises:
            InputError: If code terms are negative/nonfinite or the ledger is invalid.

        Notes:
            O(B + R_total) validation plus O(B) summation/temporary stack. Supplied
            tensors may carry gradients; built-in preparation records detached costs.
        """
        self._validate_metadata(inputs)
        n = inputs.n
        initial = scalar(inputs.initial_nll, "initial_nll")
        if bool(initial < 0) or (inputs.initial_count == 0 and bool(initial != 0)):
            raise InputError(
                "Initial code must be nonnegative and empty for an empty prefix"
            )
        costs = []
        for entry in inputs.entries:
            cost = scalar(entry.nll, "block_nll")
            if bool(cost < 0):
                raise InputError("Categorical prequential costs cannot be negative")
            costs.append(cost)
        following = initial.new_zeros(()) if not costs else torch.stack(costs).sum()
        total = initial + following
        if self.unit == "bits":
            initial, following, total = [
                q / math.log(2) for q in (initial, following, total)
            ]
        return ComplexityResult(
            method=type(self).__name__,
            kind="prequential_code",
            quantities={
                "total": Quantity(total, self.unit),
                "initial": Quantity(initial, self.unit),
                "predictive": Quantity(following, self.unit),
                "per_observation": Quantity(total / n, self.unit, "per_observation"),
            },
            primary_name="total",
            assumptions=(
                "Ledger costs were recorded before their label updates",
                "Training, calibration and replay use only revealed labels",
                "Declared class alphabet and coding protocol are shared with decoder",
            ),
        )
