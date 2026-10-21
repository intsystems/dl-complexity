"""Isolated forward calls with an explicit bridge to original parameter gradients."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

from ...errors import InputError


@contextmanager
def model_rng_scope(model, *, seed=None):
    r"""Restore Torch CPU and model-resident accelerator RNG after a scoped call.

    Args:
        model (nn.Module): Parameters/buffers determine CUDA and MPS state to save.
        seed (int, optional): Seed CPU and those devices, not all global devices.

    Yields:
        None: Caller executes within the saved RNG scope.

    Notes:
        Traverses parameter/buffer references, not every weight value. Python/NumPy
        RNG, arbitrary external devices and custom forward side effects are not
        sandboxed. Accelerator behavior has not been verified by the CPU test run.
    """
    tensors = list(model.parameters()) + list(model.buffers())
    devices = sorted({t.device.index for t in tensors if t.device.type == "cuda"})
    has_mps = any(t.device.type == "mps" for t in tensors)
    mps_state = torch.mps.get_rng_state() if has_mps else None
    with torch.random.fork_rng(devices=devices):
        try:
            if seed is not None:
                torch.default_generator.manual_seed(seed)
                for device in devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                if has_mps:
                    torch.mps.manual_seed(seed)
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


@contextmanager
def isolated_forward(model):
    r"""Yield eval-mode forwards on private state while retaining parameter gradients.

    Args:
        model (nn.Module): Deep-copyable model whose original state must not change.

    Yields:
        Callable: forward(inputs); tuple inputs become positional model arguments.

    Raises:
        InputError: Model cannot be deep-copied; use prepared numerical inputs.

    Notes:
        Deepcopy makes a private module; separate original-parameter clones provide
        an autograd bridge through torch.func.functional_call. Buffers are cloned.
        Setup copies O(parameters + buffers) values in addition to Python objects;
        forward/backward and retained activations cost as much as the chosen model.
        Parameter clones, not the original leaves, are visible to stateful forward.
    """
    try:
        isolated = copy.deepcopy(model)
    except Exception as exc:
        raise InputError(
            "Model cannot be isolated; supply numerical method inputs"
        ) from exc
    isolated.eval()
    params = {name: value.clone() for name, value in model.named_parameters()}
    buffers = {name: value.clone() for name, value in model.named_buffers()}

    def forward(inputs):
        r"""Call the isolated module with cloned parameter/buffer state.

        Args:
            inputs: Tensor or tuple of positional inputs expected by the model.

        Returns:
            object: Ordinary model output, possibly retaining an autograd graph.
        """
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        return torch.func.functional_call(isolated, (params, buffers), args)

    # CPU/CUDA random state is restored even when user forward raises. No hooks.
    with model_rng_scope(model):
        yield forward
