"""Shared input resolution for fitted-likelihood information criteria."""

from torch import Tensor

from dl_complexity._validation import positive_int, scalar
from dl_complexity.criteria.base import InformationCriterion
from dl_complexity.criteria.inputs import FitStatistics
from dl_complexity.errors import InputError
from dl_complexity.results import RequirementReport, RequirementStatus


class FittedCriterion(InformationCriterion[FitStatistics]):
    r"""Resolve the two input routes shared by fitted-likelihood criteria.

    Args:
        parameter_count (str or int): ``"trainable"`` counts unique trainable
            Parameter objects, ``"all"`` counts all unique parameters, and an
            integer explicitly supplies the statistical dimension k. This
            setting is used only for the context/data route, not to override
            ``FitStatistics.k``.

    Notes:
        The inherited ``check`` never calls forward or consumes data. The
        inherited ``estimate`` may explicitly evaluate likelihood through the
        context adapter, but never optimizes the supplied model. Subclasses
        set ``needs_n`` and implement their own native-scale formula.
    """

    input_type = FitStatistics
    needs_n = False

    def __init__(self, *, parameter_count="trainable"):
        r"""Store the parameter-count policy without accessing model weights.

        Args:
            parameter_count (str or int): ``"trainable"``, ``"all"``, or a
                nonnegative integer k for the context evaluation route.

        Raises:
            InputError: If the policy is unknown, negative, or boolean.

        Notes:
            Time and additional memory are O(1) in model/data size.
        """
        valid = (
            parameter_count in ("trainable", "all")
            if isinstance(parameter_count, str)
            else (
                isinstance(parameter_count, int)
                and not isinstance(parameter_count, bool)
                and parameter_count >= 0
            )
        )
        if not valid:
            raise InputError(
                "parameter_count must be trainable, all or a nonnegative integer"
            )
        super().__init__(parameter_count=parameter_count)

    def _check_inputs(self, inputs):
        r"""Check scalar shape, parameter count and observation metadata.

        Args:
            inputs (FitStatistics): Prepared total natural log-likelihood,
                integer k, and optional observation count n.

        Returns:
            RequirementReport: READY, or MISSING_INPUTS when this criterion
            needs n but it was omitted. READY does not verify an MLE.

        Raises:
            InputError: If likelihood shape/dtype or integer metadata is invalid.

        Notes:
            O(1) metadata work; no likelihood tensor values are scanned and no
            dataset iterator is consumed. Scalar finiteness is checked during
            numerical evaluation instead.
        """
        ll = inputs.log_likelihood
        if isinstance(ll, Tensor):
            if ll.ndim != 0 or not ll.is_floating_point():
                raise InputError(
                    "log_likelihood must be a scalar floating Tensor or Python number"
                )
        elif isinstance(ll, bool) or not isinstance(ll, (int, float)):
            raise InputError(
                "log_likelihood must be a scalar floating Tensor or Python number"
            )
        if isinstance(inputs.k, bool) or not isinstance(inputs.k, int) or inputs.k < 0:
            raise InputError("k must be a nonnegative integer fitted parameter count")
        if inputs.n is not None:
            positive_int(inputs.n, "n")
        if self.needs_n and inputs.n is None:
            return RequirementReport(RequirementStatus.MISSING_INPUTS, missing=("n",))
        return RequirementReport(
            RequirementStatus.READY,
            assumptions=("MLE/regularity and artifact origin are user-declared.",),
        )

    def _check_context(self, context, data):
        r"""Check availability of a model context, likelihood and batch source.

        Args:
            context (ModelContext or None): Declared model and likelihood.
            data (DataSource or None): Source to be evaluated later.

        Returns:
            RequirementReport: READY or a named list of missing inputs.

        Notes:
            Only object attributes are inspected: O(1) work and memory. No
            inference is made about fit quality, MLE, regularity or sample iid.
        """
        missing = []
        if context is None:
            missing.append("context")
        elif context.likelihood is None:
            missing.append("likelihood")
        if data is None:
            missing.append("data")
        if missing:
            return RequirementReport(
                RequirementStatus.MISSING_INPUTS,
                missing=tuple(missing),
                actions=("Provide FitStatistics or a model, likelihood and data.",),
            )
        return RequirementReport(
            RequirementStatus.READY,
            assumptions=(
                "An evaluation of weights does not verify MLE or regularity.",
            ),
        )

    def _prepare_context(self, context, data):
        r"""Evaluate the shared PyTorch likelihood adapter on demand.

        Args:
            context (ModelContext): Ordinary model with an explicit likelihood.
            data (DataSource): Batches with explicit observation convention.

        Returns:
            FitStatistics: Total log-likelihood, selected k, observed n and
            declared provenance, in the adapter's eval-mode convention.

        Raises:
            InputError: If likelihood/data conventions or the isolated forward
                route cannot be evaluated.

        Notes:
            This is a likelihood pass, not fitting. Cost is that of
            ``prepare_fit_statistics``: model isolation plus all forward and
            likelihood operations, rather than the O(1) criterion formula.
        """
        from dl_complexity.integrations.torch.likelihood import prepare_fit_statistics

        return prepare_fit_statistics(
            context, data=data, parameter_count=self.config["parameter_count"]
        )

    def _statistics(self, inputs):
        r"""Read finite scalar statistics used by the native criterion formula.

        Args:
            inputs (FitStatistics): Total natural log-likelihood and counts.

        Returns:
            tuple: Scalar floating Tensor log-likelihood, integer k and optional n.

        Raises:
            InputError: If the scalar is nonfinite or count metadata is invalid.

        Notes:
            O(1) arithmetic/storage. Tensor inputs retain dtype/device/autograd;
            Python numeric inputs become float64 tensors. Reading scalar
            validation results can synchronize an accelerator.
        """
        ll = scalar(inputs.log_likelihood, "log_likelihood")
        if isinstance(inputs.k, bool) or not isinstance(inputs.k, int) or inputs.k < 0:
            raise InputError("k must be a nonnegative integer fitted parameter count")
        if inputs.n is not None:
            positive_int(inputs.n, "n")
        if self.needs_n and inputs.n is None:
            raise InputError(f"{type(self).__name__} requires n observations")
        return ll, inputs.k, inputs.n
