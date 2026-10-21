"""Small protocols for model transformations and self-contained model codes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from torch import nn


@dataclass(frozen=True)
class EncodedModel:
    r"""An actual binary message, including its decoder/architecture metadata.

    Args:
        payload (bytes): Immutable, complete versioned message.

    Notes:
        Wrapping a bytes object is O(1); the message owns O(message bytes) storage.
    """

    payload: bytes

    def __post_init__(self):
        r"""Check that the message is immutable bytes in O(1).

        Raises:
            ValueError: If payload is not a bytes object.
        """
        if not isinstance(self.payload, bytes):
            raise ValueError("a model code must be immutable bytes")

    @property
    def length_bits(self) -> int:
        r"""Physical bit count, including byte padding and all headers.

        Returns:
            int: Exactly 8 * len(payload), computed in O(1) without parsing the message.
        """
        return 8 * len(self.payload)


class ModelTransform(ABC):
    """A declared, non-mutating transformation of a PyTorch model."""

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        r"""Return an independent transformed model, leaving ``model`` unchanged.

        Args:
            model (nn.Module): Source predictor, not modified by the transform.

        Returns:
            nn.Module: Independent transformed model. Concrete transforms document
            their numerical approximation, device support and O(P) copy cost.
        """


class ModelCodec(ABC):
    """Encode/decode a predictor under an explicitly specified coding language."""

    @abstractmethod
    def encode(self, model: nn.Module) -> EncodedModel:
        r"""Return a complete binary description, not a ``torch.save`` proxy.

        Args:
            model (nn.Module): Predictor whose permitted structure and state are encoded.

        Returns:
            EncodedModel: Complete model message; code length includes metadata.
        """

    @abstractmethod
    def decode(
        self,
        encoded: EncodedModel,
        model_factory: Callable[[], nn.Module] | None = None,
    ) -> nn.Module:
        r"""Reconstruct a predictor using only the message and shared codec rules.

        Args:
            encoded (EncodedModel): Message from a declared coding language.
            model_factory (callable, optional): Convenience factory, not free side information.

        Returns:
            nn.Module: Predictor reconstructed from charged bytes and shared codec rules.
        """
