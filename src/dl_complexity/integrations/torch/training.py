"""Run an explicitly attached recipe, separately from estimate and telemetry."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from ...errors import InputError
from ...context import TrainingRecord
from .model import model_rng_scope


@dataclass(frozen=True)
class TrainingRun:
    r"""Return one explicitly trained model and the callable's actual return value.

    Args:
        model (nn.Module): Independent model copy modified by the recipe.
        returned: Original callable result, not automatically detached or serialized.
        objective (str): Declared objective, not verified by tracing Python code.
        config (Mapping): Recipe arguments, including a distinct callable seed if any.
        rng_seed (int): Outer Torch RNG seed.
        status (str): Completed for returned successful runs.

    Notes:
        Unlike context.runs metadata, this returned object can retain arbitrary
        tensors/graphs through ``returned``. The caller controls their lifetime.
    """

    model: Any
    returned: Any
    objective: str
    config: Mapping[str, Any]
    rng_seed: int
    status: str = "completed"


def run_training(context, *, seed: int):
    r"""Execute the attached recipe explicitly on an independent model copy.

    Args:
        context (ModelContext): Must have a TrainingRecipe.
        seed (int): Outer Torch RNG seed, separate from recipe.config['seed'].

    Returns:
        TrainingRun: Trained copy, callable return value and declared settings.

    Raises:
        InputError: Recipe missing.
        Exception: Original training error, re-raised after recording failure.

    Notes:
        Calls recipe.train(copy, **config), restoring scoped Torch RNG. Appends only
        TrainingRecord metadata to context.runs; source model remains unchanged.
        Work is O(model copy) plus user training cost; optimizer/data policy belongs
        to the callable. No telemetry or proof of the declared objective is added.
    """
    if context.training is None:
        raise InputError("Attach a TrainingRecipe before explicitly running training")
    model = copy.deepcopy(context.model)
    recipe = context.training

    def record(status):
        r"""Append shallow run metadata without retaining the trained model or loss graph.

        Args:
            status (str): Completed or failed status for this explicit invocation.
        """
        context.runs.append(
            TrainingRecord(
                recipe.objective,
                dict(recipe.config),
                seed,
                status,
                context.state_id,
                context.likelihood_id,
            )
        )

    try:
        with model_rng_scope(model, seed=seed):
            returned = recipe.train(model, **dict(recipe.config))
    except Exception:
        record("failed")
        raise
    record("completed")
    return TrainingRun(model, returned, recipe.objective, dict(recipe.config), seed)
