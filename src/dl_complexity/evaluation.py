"""Named composition of estimators, not a benchmark or an automatic ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import ComplexityEstimator
from .data import DataSource
from .errors import InputError
from .results import ComplexityResult, RequirementReport, RequirementStatus


@dataclass(frozen=True)
class EvaluationOutcome:
    r"""Retain a result, preflight report or ordinary runtime error for one name.

    Args:
        result (ComplexityResult, optional): Successful evaluation.
        requirements (RequirementReport, optional): Preflight status.
        error (Exception, optional): Failure retained when errors='collect'.

    Notes:
        An unavailable score is not replaced by zero or silently dropped.
    """

    result: ComplexityResult | None = None
    requirements: RequirementReport | None = None
    error: Exception | None = None


class ComplexitySuite:
    r"""Compose named estimators while preserving heterogeneous result meanings.

    Args:
        methods (Mapping[str, ComplexityEstimator]): Explicit method selection.
        errors (str): 'raise' by default, or 'collect' for per-name outcomes.

    Notes:
        No training scheduler, parallel backend, automatic ranking or global cache.
    """

    def __init__(self, methods: Mapping[str, ComplexityEstimator], *, errors="raise"):
        r"""Validate and shallow-copy named methods and failure policy.

        Raises:
            InputError: Empty selection, non-estimator object or unknown error policy.

        Notes:
            O(number of methods) work/storage; no model/data computation.
        """
        if not methods or any(
            not isinstance(m, ComplexityEstimator) for m in methods.values()
        ):
            raise InputError(
                "Provide a nonempty named mapping of complexity estimators"
            )
        if errors not in {"raise", "collect"}:
            raise InputError("errors must be 'raise' or 'collect'")
        self.methods = dict(methods)
        self.errors = errors

    def _validate_names(self, inputs):
        r"""Reject inputs addressed to method names not present in this suite.

        Args:
            inputs (Mapping): Method name to that method's prepared record.

        Notes:
            Set comparison is O(number of input names + number of selected methods).
        """
        if set(inputs) - set(self.methods):
            raise InputError(
                f"Unknown method input names: {set(inputs) - set(self.methods)}"
            )

    def check(self, context=None, *, data=None, inputs=None):
        r"""Run every method's metadata preflight without consuming shared data.

        Args:
            context (ModelContext, optional): Shared model wrapper.
            data (DataSource, optional): Shared source metadata.
            inputs (Mapping, optional): Name to typed prepared input.

        Returns:
            dict[str, RequirementReport]: Report for every selected method.

        Notes:
            In collect mode InputError/TypeError become UNSUPPORTED reports. Unknown
            input names still fail. Cost is the sum of family checks, not model forward.
        """
        inputs = {} if inputs is None else inputs
        self._validate_names(inputs)
        reports = {}
        for name, method in self.methods.items():
            try:
                reports[name] = method.check(
                    context, data=data, inputs=inputs.get(name)
                )
            except (InputError, TypeError) as exc:
                if self.errors == "raise":
                    raise
                reports[name] = RequirementReport(
                    RequirementStatus.UNSUPPORTED, missing=(str(exc),)
                )
        return reports

    def estimate(self, context=None, *, data=None, inputs=None):
        r"""Preflight all methods, then evaluate selected methods sequentially.

        Args:
            context (ModelContext, optional): Shared context.
            data (DataSource, optional): Reusable source when multiple readers need it.
            inputs (Mapping, optional): Prepared inputs indexed by method name.

        Returns:
            dict[str, EvaluationOutcome]: One named outcome, including missing inputs.

        Raises:
            InputError: Failed strict preflight or multiple readers of one-shot data.

        Notes:
            Cost is the sum of method evaluations; shared contexts do not automatically
            share likelihood passes. Reuse one FitStatistics explicitly to avoid them.
            Returned results may keep autograd graphs alive. Collect mode catches
            ordinary Exceptions; strict execution has no transactional rollback.
        """
        inputs = {} if inputs is None else inputs
        reports = self.check(context, data=data, inputs=inputs)
        # Two methods cannot consume the same one-shot source without changing
        # the mathematical dataset. Do not materialize it implicitly.
        readers = [
            n
            for n, r in reports.items()
            if r.ready and n not in inputs and self.methods[n].uses_data
        ]
        if isinstance(data, DataSource) and data.one_shot and len(readers) > 1:
            raise InputError(
                "Suite needs a reusable batch source or prepared method inputs"
            )
        if self.errors == "raise" and any(not r.ready for r in reports.values()):
            raise InputError(f"Suite preflight failed: {reports}")
        outcomes = {}
        for name, method in self.methods.items():
            report = reports[name]
            if not report.ready:
                outcomes[name] = EvaluationOutcome(requirements=report)
                continue
            try:
                outcomes[name] = EvaluationOutcome(
                    result=method.estimate(context, data=data, inputs=inputs.get(name)),
                    requirements=report,
                )
            except Exception as exc:
                if self.errors == "raise":
                    raise
                outcomes[name] = EvaluationOutcome(requirements=report, error=exc)
        return outcomes
