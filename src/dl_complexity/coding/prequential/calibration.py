"""Temperature scaling for causal predictive coding.

The caller fits only on already revealed labels. Stream preparation below
scores a minibatch first and updates temperature afterwards. This is a small
past-only temperature component, not a reproduction of every EMA/replay choice
in Bornschein et al. (2022), Appendix B.1, arXiv:2210.07931.
"""

import torch
from torch.nn import functional as F

from ..._validation import as_tensor, positive_int, scalar
from ...errors import InputError


class TemperatureScaler:
    r"""Fit a positive scalar temperature on already-revealed categorical labels.

    Args:
        temperature (float or Tensor): Positive initial scalar; copied and detached.
        steps (int): Adam steps per explicit update.
        learning_rate (float): Positive finite optimizer learning rate.

    Notes:
        log_probs uses log_softmax(logits / T). update optimizes log(T) with fixed
        logits, so it cannot train the network. No observations are retained.
    """

    def __init__(
        self,
        temperature: float | torch.Tensor = 1.0,
        *,
        steps: int = 30,
        learning_rate: float = 0.05,
    ):
        r"""Validate and store a detached positive temperature in O(1).

        Args:
            temperature (float or Tensor): Positive finite scalar.
            steps (int): Positive number of calibration optimizer steps.
            learning_rate (float): Positive finite Adam rate.

        Raises:
            InputError: If temperature or optimizer configuration is invalid.
        """
        temperature = scalar(temperature, "temperature")
        if bool(temperature <= 0):
            raise InputError("Temperature must be positive")
        positive_int(steps, "steps")
        if not 0 < learning_rate < float("inf"):
            raise InputError("learning_rate must be positive and finite")
        self.temperature = temperature.detach().clone()
        self.steps = steps
        self.learning_rate = learning_rate

    def log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        r"""Compute normalized log probabilities using the current temperature.

        Args:
            logits (Tensor): Finite floating logits of shape [batch, classes >= 2].

        Returns:
            Tensor: Natural-log probabilities with unchanged shape, dtype and device.

        Raises:
            InputError: If logits are nonfinite or incorrectly shaped.

        Notes:
            O(batch * classes) time and storage. Gradients may flow into supplied logits;
            the stored temperature is detached from any previous calibration graph.
        """
        logits = as_tensor(logits)
        if (
            logits.ndim != 2
            or logits.shape[1] < 2
            or not bool(torch.isfinite(logits).all())
        ):
            raise InputError("Expected finite [observations, classes] logits")
        return F.log_softmax(logits / self.temperature.to(logits), dim=-1)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        r"""Fit temperature on this already-scored batch; store no event history.

        Args:
            logits (Tensor): Already-scored finite logits [batch, classes], detached here.
            targets (LongTensor): Revealed labels [batch] on the logits' device.

        Returns:
            TemperatureScaler: This instance, with a newly fitted detached temperature.

        Raises:
            InputError: If inputs are invalid or temperature fitting diverges.

        Notes:
            For A optimizer steps the work is O(A * batch * classes), with
            O(batch * classes) intermediates and O(1) optimizer parameter state.
            ``fit`` is an alias for this method, not a second fitting implementation.
        """
        logits = as_tensor(logits).detach()
        if logits.ndim != 2 or len(logits) == 0 or logits.shape[1] < 2:
            raise InputError(
                "Calibration needs a nonempty [observations, classes] matrix"
            )
        if not bool(torch.isfinite(logits).all()):
            raise InputError("Calibration logits must be finite")
        if targets.shape != (len(logits),) or targets.dtype != torch.long:
            raise InputError("Calibration targets must be a one-dimensional LongTensor")
        if bool(((targets < 0) | (targets >= logits.shape[1])).any()):
            raise InputError("Calibration target out of range")
        # Log-temperature enforces positivity without clipping a learned gradient.
        log_temperature = (
            self.temperature.to(logits).log().detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([log_temperature], lr=self.learning_rate)
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad()
                loss = F.cross_entropy(logits / log_temperature.exp(), targets)
                loss.backward()
                optimizer.step()
        fitted = log_temperature.exp().detach()
        if not bool(torch.isfinite(fitted)) or bool(fitted <= 0):
            raise InputError("Temperature fitting diverged")
        self.temperature = fitted
        return self

    fit = update
