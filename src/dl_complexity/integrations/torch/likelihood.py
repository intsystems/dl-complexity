"""Likelihood adapters make reduction explicit; arbitrary losses are not densities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..._validation import positive_int
from ...data import DataSource
from ...errors import InputError
from ...results import Provenance
from .model import isolated_forward


@dataclass(frozen=True)
class CategoricalLikelihood:
    r"""Use a normalized categorical likelihood for one class label per row.

    Notes:
        Input logits have shape [N,C]; targets have shape [N] and dtype long.
        Cross-entropy weights, soft labels and smoothing are not silently assumed.
    """

    def log_prob(self, prediction, target):
        r"""Compute log_softmax(logits)[row, true_class] without batch reduction.

        Args:
            prediction (torch.Tensor): Floating logits [N,C].
            target (torch.Tensor): Long class indices [N].

        Returns:
            torch.Tensor: Natural-log probabilities [N] with prediction gradients.

        Raises:
            InputError: Invalid shapes or target dtype.

        Notes:
            O(N*C) time and log-softmax storage; gather returns N selected entries.
            Normalizing via log_softmax avoids exponentiating unshifted logits.
        """
        if prediction.ndim != 2 or target.shape != prediction.shape[:1]:
            raise InputError(
                "Categorical likelihood expects logits [N,C] and targets [N]"
            )
        if target.dtype != torch.long:
            raise InputError("Categorical targets must have dtype torch.long")
        return F.log_softmax(prediction, dim=-1).gather(-1, target[:, None]).squeeze(-1)


@dataclass(frozen=True)
class GaussianLikelihood:
    r"""Use independent Gaussian event coordinates with fixed known noise scale.

    Args:
        sigma (float): Positive standard deviation, shared by all coordinates.

    Notes:
        Includes normalization constants; MSE alone does not supply this density.
    """

    sigma: float = 1.0

    def __post_init__(self):
        r"""Reject nonfinite/nonpositive fixed noise scale before evaluation."""
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise InputError("sigma must be finite and positive")

    def log_prob(self, prediction, target):
        r"""Evaluate Normal log density and sum event axes, preserving observation axis.

        Args:
            prediction (torch.Tensor): Mean values [N,...].
            target (torch.Tensor): Observations with exactly the same shape.

        Returns:
            torch.Tensor: Total event log density per observation, shape [N], in nats.

        Raises:
            InputError: Missing observation axis or incompatible shapes.

        Notes:
            Per coordinate: -0.5*((y-mean)/sigma)**2-log(sigma)-0.5*log(2*pi).
            O(number of event coordinates) work/storage; no silent broadcasting.
            A density NLL may be negative and is not automatically a discrete code.
        """
        if prediction.shape != target.shape or prediction.ndim < 1:
            raise InputError(
                "Gaussian prediction and target must have equal [N,...] shape"
            )
        pointwise = -0.5 * ((target - prediction) / self.sigma).square()
        pointwise = pointwise - math.log(self.sigma) - 0.5 * math.log(2 * math.pi)
        return pointwise.reshape(len(prediction), -1).sum(-1)


def prepare_fit_statistics(context, *, data: DataSource, parameter_count="trainable"):
    r"""Compute total log likelihood, declared parameter dimension and observed n.

    Args:
        context (ModelContext): Model and explicit likelihood; loss alone is insufficient.
        data (DataSource): Batches with pointwise log_prob output [N_batch].
        parameter_count (str or int): 'trainable', 'all' or explicit nonnegative k.

    Returns:
        FitStatistics: Total natural log likelihood, k, n and declared provenance.

    Raises:
        InputError: Missing likelihood, wrong batch/reduction, nonfinite likelihood
            or observed n disagreeing with declared n.

    Notes:
        Counts unique Parameter objects, not effective statistical degrees of
        freedom; no optimization or MLE claim. Evaluates an isolated eval-mode copy,
        sums log_prob over observations and validates n after consumption.
        Time includes copying model state plus all forwards/likelihood reductions.
        Grad mode is preserved: enabled autograd can retain every batch's graph,
        not only the last batch. Use torch.no_grad() explicitly when appropriate.
    """
    from ...criteria.inputs import FitStatistics

    if context is None or context.likelihood is None:
        raise InputError(
            "An explicit likelihood is required; loss alone is insufficient"
        )
    if not isinstance(data, DataSource):
        raise InputError("Provide a DataSource with explicit observation conventions")
    if parameter_count in ("trainable", "all"):
        k = sum(
            p.numel()
            for p in context.model.parameters()
            if parameter_count == "all" or p.requires_grad
        )
    elif (
        isinstance(parameter_count, int)
        and not isinstance(parameter_count, bool)
        and parameter_count >= 0
    ):
        k = parameter_count
    else:
        raise InputError(
            "parameter_count must be 'trainable', 'all' or a nonnegative integer"
        )
    total, n = None, 0
    log_prob = getattr(context.likelihood, "log_prob", context.likelihood)
    with isolated_forward(context.model) as forward:
        for batch in data:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise InputError("Each batch must contain (inputs, targets)")
            inputs, targets = batch
            values = log_prob(forward(inputs), targets)
            if not isinstance(values, torch.Tensor) or values.ndim != 1:
                raise InputError(
                    "likelihood must return pointwise Tensor [N], not a reduced loss"
                )
            if (
                targets.ndim == 0
                or len(values) != len(targets)
                or not bool(torch.isfinite(values).all())
            ):
                raise InputError("Invalid observation axis or nonfinite log likelihood")
            total = values.sum() if total is None else total + values.sum()
            n += len(values)
    positive_int(n, "observed n")
    if data.n_observations is not None and n != data.n_observations:
        raise InputError("Observed n does not match DataSource.n_observations")
    return FitStatistics(
        total, k, n, Provenance(context.state_id, data.data_id, context.likelihood_id)
    )
