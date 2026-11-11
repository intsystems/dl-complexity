"""Record independent analytic oracles alongside ordinary pytest assertions."""

import json
import math

import pytest
import torch


@pytest.fixture
def record_analytic(record_property, request):
    """Compare one scalar with its oracle and attach a machine-readable record.

    JUnit properties are evidence from the running tests, not hard-coded PASS
    labels. A failed comparison is recorded before the assertion so the report
    can retain failures. Use ``-o junit_family=xunit1`` for property-compatible XML.
    """

    def compare(
        *,
        case_id,
        method,
        model,
        expected,
        actual,
        unit,
        oracle,
        atol=1e-10,
        rtol=1e-10,
        category="exact",
        details=None,
    ):
        """Record finite observed/oracle scalars before checking their tolerance."""
        if isinstance(actual, torch.Tensor):
            assert actual.ndim == 0, "An analytic row must describe a scalar"
            actual = actual.detach().cpu().item()
        actual, expected = float(actual), float(expected)
        assert category in {"exact", "finite_draw", "approximation"}
        assert math.isfinite(atol) and math.isfinite(rtol) and min(atol, rtol) >= 0
        passed = (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol)
        )
        row = {
            "case_id": case_id,
            "test": request.node.nodeid,
            "method": method,
            "model": model,
            "expected": expected,
            "actual": actual,
            "absolute_error": abs(actual - expected),
            "unit": unit,
            "oracle": oracle,
            "atol": atol,
            "rtol": rtol,
            "category": category,
            "details": details or {},
            "passed": passed,
        }
        record_property("analytic_case", json.dumps(row, ensure_ascii=False))
        assert passed, (
            f"{case_id}: actual={actual!r}, expected={expected!r}, "
            f"atol={atol}, rtol={rtol}; {oracle}"
        )
        return actual

    return compare
