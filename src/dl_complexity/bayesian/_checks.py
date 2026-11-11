"""Metadata-only preflight helpers: no factorisation, forward or tensor scans."""

from torch import Tensor

from dl_complexity.errors import InputError


def scalar_metadata(value, name):
    r"""Validate scalar type/shape without inspecting Tensor contents.

    Args:
        value (float or Tensor): Python int/float or zero-dimensional floating
            Tensor. Boolean inputs are deliberately not numeric statistics.
        name (str): Field label used in an error message.

    Raises:
        InputError: If the type, tensor rank or dtype is incompatible.

    Notes:
        O(1) time/storage; no finite-value scan, device synchronization or
        differentiation is needed. Numeric finiteness is an estimate-time check.
    """
    if isinstance(value, Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise InputError(
                f"{name} must be a scalar floating Tensor or Python number"
            )
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a scalar floating Tensor or Python number")
