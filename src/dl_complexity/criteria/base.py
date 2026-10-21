"""Information criteria share conventions, not an obligatory training phase."""

from typing import Generic, TypeVar

from dl_complexity.base import ComplexityEstimator

InputsT = TypeVar("InputsT")


class InformationCriterion(ComplexityEstimator[InputsT], Generic[InputsT]):
    r"""Base interface for information criteria with explicit native scale.

    Notes:
        Generic inputs distinguish fitted-likelihood summaries from posterior
        likelihood matrices. This class adds no fitting state or numeric work;
        ``check`` and ``estimate`` are inherited from ComplexityEstimator.
        Do not infer sklearn ``fit``/Pipeline compatibility from this hierarchy.
    """
