"""Composition around an ordinary PyTorch model, not a new training framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn

from .errors import InputError


@dataclass(frozen=True)
class TrainingRecipe:
    r"""Declare a training callable and configuration without executing it.

    Args:
        train (Callable): Called as train(model, **config) by run_training.
        objective (str): Human-readable declaration, not inferred from Python code.
        config (Mapping): Optimizer/data/other arguments owned by the callable.

    Notes:
        A frozen record is shallow: it neither freezes tensors nor serializes closures.
    """

    train: Callable
    objective: str
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRecord:
    r"""Store executed-run metadata without retaining a trained model or loss graph.

    Args:
        objective (str): Declared objective.
        config (Mapping): Shallow recipe configuration snapshot.
        rng_seed (int): Outer PyTorch RNG seed, distinct from a callable's seed arg.
        status (str): Completed or failed for recorded ordinary exceptions.
        source_state_id (str, optional): User-declared starting model identity.
        likelihood_id (str, optional): User-declared likelihood identity.
    """

    objective: str
    config: Mapping[str, Any]
    rng_seed: int
    status: str
    source_state_id: str | None = None
    likelihood_id: str | None = None


@dataclass(frozen=True)
class ParameterInfo:
    r"""Describe one named parameter reference, including tied-object aliases.

    Args:
        name (str): Named-parameter path.
        shape (tuple[int, ...]): Tensor dimensions, without copying values.
        trainable (bool): Current requires_grad flag.
        alias_of (str, optional): First name referring to the same Parameter object.
    """

    name: str
    shape: tuple[int, ...]
    trainable: bool
    alias_of: str | None


@dataclass(frozen=True)
class ModelStructure:
    r"""Represent named modules and parameter references, not a forward graph.

    Args:
        modules (tuple): Pairs of module path and class name.
        parameters (tuple[ParameterInfo, ...]): Parameter-reference metadata.

    Notes:
        No forward execution, FX/export tracing or persistent autograd graph.
    """

    modules: tuple[tuple[str, str], ...]
    parameters: tuple[ParameterInfo, ...]


@dataclass
class ModelContext:
    r"""Compose an ordinary nn.Module with likelihood, loss and training metadata.

    Args:
        model (nn.Module): Existing model, not copied by this constructor.
        loss (Callable, optional): User loss; never assumed to define a density.
        likelihood (object, optional): Callable or object exposing log_prob(pred, y).
        training (TrainingRecipe, optional): Recipe, not an automatic fit request.
        state_id (str, optional): Declared model identity, not a weight checksum.
        likelihood_id (str, optional): Declared likelihood identity.
        loss_likelihood_relation (str): Description, not a verified algebraic mapping.
        runs (list[TrainingRecord]): Metadata of explicit run_training calls.

    Notes:
        Constructing the wrapper does not train, trace or inspect parameter values.
    """

    model: nn.Module
    loss: Callable | None = None
    likelihood: Any | None = None
    training: TrainingRecipe | None = None
    state_id: str | None = None
    likelihood_id: str | None = None
    loss_likelihood_relation: str = "unspecified"
    runs: list[TrainingRecord] = field(default_factory=list)

    def __post_init__(self):
        r"""Require an nn.Module without inspecting its forward or weights."""
        if not isinstance(self.model, nn.Module):
            raise InputError("ModelContext requires an nn.Module")

    @property
    def structure(self) -> ModelStructure:
        r"""Recompute module/parameter metadata and tied-object aliases.

        Returns:
            ModelStructure: Named modules and every named parameter reference.

        Notes:
            Uses id(parameter), not storage overlap, for aliases. Time/memory scale
            with module and parameter-reference counts, not the number of weight values.
        """
        aliases = {}
        parameters = []
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            previous = aliases.setdefault(id(parameter), name)
            parameters.append(
                ParameterInfo(
                    name,
                    tuple(parameter.shape),
                    parameter.requires_grad,
                    previous if previous != name else None,
                )
            )
        return ModelStructure(
            tuple((n, type(m).__qualname__) for n, m in self.model.named_modules()),
            tuple(parameters),
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        model_factory: Callable[[], nn.Module],
        map_location="cpu",
        **kwargs,
    ):
        r"""Load a plain weights-only state_dict using a caller-supplied architecture.

        Args:
            path (str or Path): Ordinary state_dict checkpoint, not a pickled module.
            model_factory (Callable): Builds matching architecture and parameter dtypes.
            map_location: Passed to torch.load; factory controls destination modules.
            **kwargs: ModelContext fields such as likelihood and declared state_id.

        Returns:
            ModelContext: Fresh model with strict-loaded checkpoint tensors.

        Raises:
            InputError: Not a state mapping or factory dtype would silently cast values.
            RuntimeError: PyTorch rejects missing, extra or shape-mismatched state keys.

        Notes:
            Deserialize/copy cost is proportional to checkpoint bytes, in addition to
            model_factory work. An architecture cannot be recovered from state_dict alone.
            weights_only reduces pickle exposure; do not treat untrusted files as safe.
        """
        model = model_factory()
        state = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(state, Mapping):
            raise InputError("Checkpoint must be a state_dict mapping")
        expected = model.state_dict()
        for name, value in state.items():
            if (
                name in expected
                and isinstance(value, torch.Tensor)
                and value.dtype != expected[name].dtype
            ):
                raise InputError(
                    f"Factory dtype for {name} differs from checkpoint; provide a matching factory"
                )
        model.load_state_dict(state, strict=True)
        return cls(model=model, **kwargs)
