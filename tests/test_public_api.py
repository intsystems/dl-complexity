"""Stable exports and one estimator protocol across the mathematical families."""

import importlib

import pytest

from dl_complexity import ComplexityEstimator, ComplexityResult, RequirementReport
from dl_complexity.bayesian import KFACLaplace, VariationalCode, WBIC
from dl_complexity.bounds import AroraCompressionBound
from dl_complexity.coding import (
    BlockwisePrequential,
    OnlinePrequential,
    ReplayPrequential,
    TwoPartMDL,
)
from dl_complexity.criteria import AIC, BIC, HQIC, WAIC
from dl_complexity.diagnostics import WeightStatistics


@pytest.mark.parametrize(
    "module_name",
    [
        "dl_complexity",
        "dl_complexity.criteria",
        "dl_complexity.bayesian",
        "dl_complexity.coding",
        "dl_complexity.compression",
        "dl_complexity.bounds",
        "dl_complexity.diagnostics",
        "dl_complexity.integrations.torch",
    ],
)
def test_public_exports_exist_and_are_unique(module_name):
    module = importlib.import_module(module_name)
    assert len(module.__all__) == len(set(module.__all__))
    assert all(
        isinstance(name, str) and hasattr(module, name) for name in module.__all__
    )


@pytest.mark.parametrize(
    "method",
    [
        AIC(),
        BIC(),
        HQIC(),
        WAIC(),
        WBIC(),
        VariationalCode(),
        KFACLaplace(),
        TwoPartMDL(),
        BlockwisePrequential(),
        OnlinePrequential(),
        ReplayPrequential(),
        AroraCompressionBound(),
        WeightStatistics(),
    ],
)
def test_same_public_protocol_and_no_fictitious_fit(method):
    assert isinstance(method, ComplexityEstimator)
    assert type(method).estimate is ComplexityEstimator.estimate
    assert type(method).check is ComplexityEstimator.check
    assert isinstance(method.check(), RequirementReport)
    assert not hasattr(method, "fit")
