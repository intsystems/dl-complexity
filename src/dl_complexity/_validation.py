"""Small tensor checks shared by formulas, not a second numeric backend."""

from __future__ import annotations

import torch

from .errors import InputError


def as_tensor(value) -> torch.Tensor:
    r"""Preserve a floating tensor or construct a float64 tensor from Python values.

    Args:
        value: Floating Tensor or values accepted by torch.as_tensor.

    Returns:
        torch.Tensor: Existing tensor unchanged, including dtype/device/autograd.

    Raises:
        InputError: Tensor input has a non-floating dtype.

    Notes:
        Existing-tensor route is O(1); constructing m values needs O(m) storage.
    """
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise InputError("Expected a floating-point tensor")
        return value
    return torch.as_tensor(value, dtype=torch.float64)


def scalar(value, name: str = "value") -> torch.Tensor:
    r"""Require a finite, zero-dimensional floating tensor.

    Args:
        value: Scalar tensor or Python scalar convertible through as_tensor.
        name (str): Field label used in the validation error.

    Returns:
        torch.Tensor: Scalar without detaching its gradient graph.

    Raises:
        InputError: Non-scalar shape or nonfinite value.

    Notes:
        Constant scalar work; converting the finite predicate to bool can
        synchronize an accelerator with the host.
    """
    value = as_tensor(value)
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise InputError(f"{name} must be a finite scalar")
    return value


def positive_int(value, name: str = "value") -> int:
    r"""Validate a positive Python integer while rejecting booleans.

    Args:
        value: Declared count; tensor integers are not silently unwrapped.
        name (str): Field label for errors.

    Returns:
        int: Unchanged positive integer.

    Raises:
        InputError: Non-integer, bool, zero or negative count.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer")
    return value
