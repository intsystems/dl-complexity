"""One public evaluation protocol for every complexity estimator family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from time import perf_counter
from typing import Generic, TypeVar

from .errors import InputError, PreparationRequired
from .results import ComplexityResult, RequirementReport, RequirementStatus

TInputs = TypeVar("TInputs")


class ComplexityEstimator(ABC, Generic[TInputs]):
    r"""Define the shared typed evaluation protocol, without an obligatory fit.

    Args:
        **config: Constructor settings recorded in every result.

    Attributes:
        input_type (type or tuple[type]): Accepted prepared-input record classes.
        uses_data (bool): Whether the context route may consume a data source.

    Notes:
        Subclasses implement ``_estimate`` and optionally a direct context route.
        Posterior sampling, optimization and curvature collection remain explicit
        preparation functions, never accidental effects of ``check``.
    """

    input_type: type | tuple[type, ...]
    uses_data = True

    def __init__(self, **config):
        r"""Copy configuration keys without fitting or inspecting data.

        Args:
            **config: Method settings; values are not deep-copied.

        Notes:
            Time and extra memory are O(number of configuration entries).
        """
        self._config = dict(config)

    @property
    def config(self):
        r"""Return a shallow copy of constructor configuration.

        Returns:
            dict: Independent key mapping; nested values may still be shared.
        """
        return dict(getattr(self, "_config", {}))

    def check(self, context=None, *, data=None, inputs=None) -> RequirementReport:
        r"""Check input metadata and declared identities without executing a model.

        Args:
            context (ModelContext, optional): Model and likelihood declarations.
            data (DataSource, optional): Observation source, not consumed here.
            inputs (object, optional): This method's prepared numerical record.

        Returns:
            RequirementReport: Computability, missing information and assumptions.

        Raises:
            InputError: Wrong input type or conflicting nonempty provenance IDs.

        Notes:
            Dispatches to family metadata checks. READY is not evidence of MLE/MAP,
            iid sampling or convergence; unknown identities are not content hashes.
            No forward, training, sampling, RNG mutation or iterator consumption.
        """
        if inputs is None:
            return self._check_context(context, data)
        if not isinstance(inputs, self.input_type):
            raise InputError(
                f"{type(self).__name__} expects {self.input_type}, got {type(inputs)}"
            )
        provenance = getattr(inputs, "provenance", None)
        if provenance is not None:
            identities = (
                ("state_id", context),
                ("likelihood_id", context),
                ("data_id", data),
            )
            for name, source in identities:
                declared, actual = getattr(provenance, name, None), getattr(
                    source, name, None
                )
                if declared is not None and actual is not None and declared != actual:
                    raise InputError(
                        f"Contradicting {name}: {declared!r} != {actual!r}"
                    )
        return self._check_inputs(inputs)

    def _check_inputs(self, inputs) -> RequirementReport:
        r"""Accept a correctly typed record, leaving family-specific checks to overrides.

        Returns:
            RequirementReport: READY with an explicit user-declared-provenance caveat.

        Notes:
            Constant work; no tensor reduction is performed by this base method.
        """
        return RequirementReport(
            RequirementStatus.READY,
            assumptions=("Artifact provenance and applicability are user-declared",),
        )

    def _check_context(self, context, data) -> RequirementReport:
        r"""Describe the required prepared record when no direct adapter is defined.

        Returns:
            RequirementReport: MISSING_INPUTS listing accepted record class names.

        Notes:
            O(number of accepted record types), without inspecting model weights.
        """
        types = (
            self.input_type
            if isinstance(self.input_type, tuple)
            else (self.input_type,)
        )
        return RequirementReport(
            RequirementStatus.MISSING_INPUTS,
            missing=(" or ".join(t.__name__ for t in types),),
            actions=("Provide the method's explicit prepared inputs",),
        )

    def _prepare_context(self, context, data):
        r"""Reject implicit preparation unless a concrete direct adapter overrides it.

        Raises:
            PreparationRequired: This base implementation cannot manufacture artifacts.
        """
        raise PreparationRequired("Explicit preparation is required")

    def estimate(self, context=None, *, data=None, inputs=None) -> ComplexityResult:
        r"""Validate, resolve direct inputs and evaluate one mathematical quantity.

        Args:
            context (ModelContext, optional): Ordinary PyTorch model wrapper.
            data (DataSource, optional): Data for a supported direct evaluation route.
            inputs (object, optional): Sufficient prepared values; bypass model forward.

        Returns:
            ComplexityResult: Named quantities, settings, provenance and elapsed time.

        Raises:
            InputError: Invalid or missing mathematical information.
            PreparationRequired: A declared preparation action must be called first.

        Notes:
            Calls check, optionally _prepare_context, then _estimate. Result-specific
            configuration overrides duplicate constructor keys. Time excludes preflight
            and is a host perf_counter interval, not a synchronized GPU benchmark.
            Wrapper cost is constant apart from copied configuration and family work.
        """
        report = self.check(context, data=data, inputs=inputs)
        if not report.ready:
            error = (
                PreparationRequired
                if report.status == RequirementStatus.PREPARATION_REQUIRED
                else InputError
            )
            raise error(
                f"{report.status.value}: missing={report.missing}; actions={report.actions}"
            )
        start = perf_counter()
        prepared = self._prepare_context(context, data) if inputs is None else inputs
        result = self._estimate(prepared)
        return replace(
            result,
            config={**self.config, **result.config},
            provenance=result.provenance or getattr(prepared, "provenance", None),
            elapsed_seconds=perf_counter() - start,
        )

    @abstractmethod
    def _estimate(self, inputs: TInputs) -> ComplexityResult:
        r"""Compute the family formula from typed sufficient inputs.

        Args:
            inputs: Prepared record accepted by ``input_type``.

        Returns:
            ComplexityResult: Formula output preserving its own estimand and units.
        """
