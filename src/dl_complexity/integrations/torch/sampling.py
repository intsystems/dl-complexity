"""Langevin sampling of the posterior for WAIC and WBIC.

Samples the trainable weights w of a model from::

    p(w) ~ exp(-beta * nL_n(w) - alpha/2 |w|^2 - gamma/2 |w - w0|^2),

where w0 are the current weights. beta = 1 is the usual posterior (WAIC),
beta = 1/log n is the WBIC one (Watanabe 2013), gamma > 0 localizes the chain
around w0 (Lau et al. 2024). Step: w' = w - h grad U + sqrt(2h) xi (ULA/SGLD),
optionally with a Metropolis correction (MALA).
"""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass

import torch

from ..._validation import positive_int
from ...data import DataSource
from ...errors import InputError
from ...results import Provenance
from .model import model_rng_scope


def _finite_positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise InputError(f"{name} must be a finite positive number")
    return float(value)


def _finite_nonnegative(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise InputError(f"{name} must be a finite nonnegative number")
    return float(value)


@dataclass(frozen=True)
class Langevin:
    r"""Settings of a Langevin chain.

    Args:
        step_size (float): step h.
        num_draws (int): number of saved draws.
        burn_in (int): steps skipped before the first draw.
        thin (int): steps between saved draws.
        metropolis (bool): MALA instead of plain Langevin (full batch only).
        minibatch (bool): SGLD, one batch per step.
        seed (int, optional): random seed; a random one is used if None.

    With minibatch=True batches go in a cycle, so pick thin coprime with
    the number of batches.
    """

    step_size: float
    num_draws: int
    burn_in: int = 0
    thin: int = 1
    metropolis: bool = False
    minibatch: bool = False
    seed: int | None = None

    def __post_init__(self):
        _finite_positive(self.step_size, "step_size")
        positive_int(self.num_draws, "num_draws")
        positive_int(self.thin, "thin")
        if isinstance(self.burn_in, bool) or not isinstance(self.burn_in, int):
            raise InputError("burn_in must be a nonnegative integer")
        if self.burn_in < 0:
            raise InputError("burn_in must be a nonnegative integer")
        for name in ("metropolis", "minibatch"):
            if not isinstance(getattr(self, name), bool):
                raise InputError(f"{name} must be a boolean")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise InputError("seed must be a nonnegative integer or None")
        if self.metropolis and self.minibatch:
            raise InputError(
                "The Metropolis correction needs full-batch log densities; "
                "use minibatch=False"
            )

    @property
    def total_steps(self) -> int:
        return self.burn_in + self.num_draws * self.thin


@dataclass(frozen=True)
class PosteriorDraws:
    r"""Result of sample_posterior.

    Args:
        parameters (Tensor): draws [S, P], weights flattened in ``names`` order.
        names (tuple): names of the sampled parameters.
        log_likelihood (Tensor): log p(y_i | x_i, w_s) for all draws, [S, n].
        n (int): number of observations.
        beta, prior_precision, localization (float): the target settings.
        acceptance_rate (float or None): MALA acceptance (with burn-in).
        provenance (Provenance, optional): ids of model, data and likelihood.
        seed (int, optional): seed that was actually used.
    """

    parameters: torch.Tensor
    names: tuple
    log_likelihood: torch.Tensor
    n: int
    beta: float
    prior_precision: float
    localization: float
    acceptance_rate: float | None
    provenance: Provenance | None = None
    seed: int | None = None


class _Target:
    """U(w) = -log p(w), its gradient and the pointwise log-likelihood.

    Works on an eval-mode copy of the model, so the original is not touched.
    """

    def __init__(self, context, data, *, alpha, beta, gamma):
        try:
            model = copy.deepcopy(context.model)
        except Exception as exc:
            raise InputError(
                "Model cannot be isolated; supply numerical method inputs"
            ) from exc
        self.model = model.eval()
        named = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
        if not named:
            raise InputError("The model has no trainable parameters to sample")
        dtypes = {p.dtype for _, p in named}
        devices = {p.device for _, p in named}
        if len(dtypes) != 1 or not named[0][1].is_floating_point():
            raise InputError("Sampled parameters must share one floating dtype")
        if len(devices) != 1:
            raise InputError("Sampled parameters must live on one device")
        self.names = tuple(k for k, _ in named)
        self.shapes = [p.shape for _, p in named]
        self.sizes = [p.numel() for _, p in named]
        self.frozen = {
            k: p.detach().clone()
            for k, p in model.named_parameters()
            if not p.requires_grad
        }
        self.buffers = {k: b.detach().clone() for k, b in model.named_buffers()}
        self.start = torch.cat([p.detach().reshape(-1) for _, p in named]).clone()
        # sum U in float64 for the MALA ratio (MPS has no float64)
        self.energy_dtype = (
            torch.float32 if self.start.device.type == "mps" else torch.float64
        )
        self.data = data
        self.log_prob = getattr(context.likelihood, "log_prob", context.likelihood)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def _state(self, w):
        state, offset = dict(self.frozen), 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            state[name] = w[offset : offset + size].reshape(shape)
            offset += size
        return state

    def pointwise(self, w, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise InputError("Each batch must contain (inputs, targets)")
        inputs, targets = batch
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        prediction = torch.func.functional_call(
            self.model, (self._state(w), self.buffers), args
        )
        values = self.log_prob(prediction, targets)
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or targets.ndim == 0
            or len(values) != len(targets)
        ):
            raise InputError(
                "likelihood must return pointwise Tensor [N], not a reduced loss"
            )
        return values

    def prior_terms(self, w):
        value = 0.5 * self.alpha * w.square().sum()
        if self.gamma:
            value = value + 0.5 * self.gamma * (w - self.start).square().sum()
        return value

    def full(self, w, *, grad):
        """One pass over the data: U, grad U, pointwise log-lik, n, batch count."""
        w = w.detach().requires_grad_(grad)
        total, n, count, rows = 0.0, 0, 0, []
        gradient = torch.zeros_like(w) if grad else None
        with torch.set_grad_enabled(grad):
            for batch in self.data:
                values = self.pointwise(w, batch)
                if grad:  # backward per batch, so only one graph is kept
                    (piece,) = torch.autograd.grad(-self.beta * values.sum(), w)
                    gradient += piece
                values = values.detach()
                rows.append(values)
                total = total + values.sum(dtype=self.energy_dtype)
                n += len(values)
                count += 1
            if n == 0:
                raise InputError("The DataSource yielded no observations")
            prior = self.prior_terms(w)
            if grad:
                (piece,) = torch.autograd.grad(prior, w)
                gradient += piece
        energy = self.beta * (-total) + prior.detach().to(self.energy_dtype)
        return energy, gradient, torch.cat(rows), n, count

    def minibatch_gradient(self, w, batch, scale):
        # scale = number of batches, so over one pass every point counts once
        # (n / B would overweight a smaller last batch)
        w = w.detach().requires_grad_(True)
        with torch.enable_grad():
            values = self.pointwise(w, batch)
            energy = -self.beta * scale * values.sum() + self.prior_terms(w)
            (gradient,) = torch.autograd.grad(energy, w)
        return energy.detach(), gradient


def _check_finite(*tensors, step):
    for tensor in tensors:
        if not bool(torch.isfinite(tensor).all()):
            raise InputError(
                f"Langevin chain diverged at step {step}; reduce step_size"
            )


def sample_posterior(
    context,
    *,
    data,
    sampler,
    prior_precision,
    beta=1.0,
    localization=0.0,
):
    r"""Run a Langevin chain starting from the current weights of the model.

    Args:
        context (ModelContext): model with a pointwise likelihood.
        data (DataSource): reusable batches (inputs, targets).
        sampler (Langevin): chain settings.
        prior_precision (float): alpha in the prior N(0, I / alpha).
        beta (float): inverse temperature, 1 for WAIC, 1/log n for WBIC.
        localization (float): gamma, pull towards the start; 0 = off.

    Returns:
        PosteriorDraws

    The model itself, its mode and the global torch RNG are not changed.
    One full-batch step costs one forward/backward pass over the data.
    """
    if context is None or context.likelihood is None:
        raise InputError(
            "An explicit likelihood is required; loss alone is insufficient"
        )
    if not isinstance(data, DataSource):
        raise InputError("Provide a DataSource with explicit observation conventions")
    if data.one_shot:
        raise InputError(
            "Sampling iterates the data many times; provide a reusable DataSource"
        )
    if not isinstance(sampler, Langevin):
        raise InputError("sampler must be a Langevin configuration")
    alpha = _finite_positive(prior_precision, "prior_precision")
    beta = _finite_positive(beta, "beta")
    gamma = _finite_nonnegative(localization, "localization")

    with model_rng_scope(context.model):
        target = _Target(context, data, alpha=alpha, beta=beta, gamma=gamma)
        w = target.start.clone()
        generator = torch.Generator(device=w.device)
        if sampler.seed is None:
            seed = generator.seed()
        else:
            seed = sampler.seed
            generator.manual_seed(seed)

        energy, gradient, pointwise, n, num_batches = target.full(
            w, grad=not sampler.minibatch
        )
        if data.n_observations is not None and n != data.n_observations:
            raise InputError("Observed n does not match DataSource.n_observations")
        _check_finite(energy, step=0)

        h = sampler.step_size
        noise_sd = math.sqrt(2 * h)
        draws, rows = [], []
        accepted = 0
        batches = iter(data)

        def next_batch():
            nonlocal batches
            try:
                return next(batches)
            except StopIteration:
                batches = iter(data)
                return next(batches)

        def full_pass(point, *, grad):  # same data on every pass
            result = target.full(point, grad=grad)
            if result[3:] != (n, num_batches):
                raise InputError(
                    "The DataSource changed its number of observations or batches"
                )
            return result[:3]

        for step in range(1, sampler.total_steps + 1):
            noise = torch.randn(
                w.shape, generator=generator, dtype=w.dtype, device=w.device
            )
            if sampler.minibatch:
                _, gradient = target.minibatch_gradient(w, next_batch(), num_batches)
                w = (w - h * gradient + noise_sd * noise).detach()
                _check_finite(w, step=step)
            else:
                proposal = (w - h * gradient + noise_sd * noise).detach()
                _check_finite(proposal, step=step)
                p_energy, p_gradient, p_pointwise = full_pass(proposal, grad=True)
                if sampler.metropolis:  # MH with Gaussian proposals q(w'|w)
                    forward = (proposal - w + h * gradient).square().sum() / (4 * h)
                    backward = (w - proposal + h * p_gradient).square().sum() / (4 * h)
                    log_ratio = energy - p_energy + forward - backward
                    uniform = torch.rand(
                        (), generator=generator, dtype=w.dtype, device=w.device
                    )
                    move = bool(torch.isfinite(p_energy)) and math.log(
                        max(float(uniform), 1e-300)
                    ) < float(log_ratio)
                else:
                    _check_finite(p_energy, step=step)
                    move = True
                if move:
                    w, energy, gradient, pointwise = (
                        proposal,
                        p_energy,
                        p_gradient,
                        p_pointwise,
                    )
                    accepted += 1
            if step > sampler.burn_in and (step - sampler.burn_in) % sampler.thin == 0:
                if sampler.minibatch:
                    _, _, pointwise = full_pass(w, grad=False)
                draws.append(w.clone())
                rows.append(pointwise.clone())

    return PosteriorDraws(
        parameters=torch.stack(draws),
        names=target.names,
        log_likelihood=torch.stack(rows),
        n=n,
        beta=beta,
        prior_precision=alpha,
        localization=gamma,
        acceptance_rate=(
            accepted / sampler.total_steps if sampler.metropolis else None
        ),
        provenance=Provenance(context.state_id, data.data_id, context.likelihood_id),
        seed=seed,
    )


def prepare_posterior_log_likelihood(context, *, data, sampler, prior_precision):
    r"""Draws from the usual posterior (beta = 1) as input for WAIC.

    Needs at least two draws. Whether the chain has mixed is not checked.
    """
    from ...criteria.inputs import PosteriorLogLikelihood

    if not isinstance(sampler, Langevin):
        raise InputError("sampler must be a Langevin configuration")
    if sampler.num_draws < 2:
        raise InputError("WAIC needs at least two posterior draws")
    draws = sample_posterior(
        context, data=data, sampler=sampler, prior_precision=prior_precision
    )
    return PosteriorLogLikelihood(draws.log_likelihood, provenance=draws.provenance)


def prepare_tempered_likelihood(context, *, data, sampler, prior_precision):
    r"""Draws at beta = 1/log n as input for WBIC.

    If the DataSource does not declare n, it is counted first.
    """
    from ...bayesian.inputs import TemperedLikelihood

    if not isinstance(data, DataSource):
        raise InputError("Provide a DataSource with explicit observation conventions")
    if data.one_shot:
        raise InputError(
            "Sampling iterates the data many times; provide a reusable DataSource"
        )
    n = data.n_observations
    if n is None:
        n = 0
        for batch in data:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise InputError("Each batch must contain (inputs, targets)")
            n += len(batch[1])
    if n <= 1:
        raise InputError("WBIC requires n > 1")
    draws = sample_posterior(
        context,
        data=data,
        sampler=sampler,
        prior_precision=prior_precision,
        beta=1 / math.log(n),
    )
    return TemperedLikelihood(
        draws.log_likelihood.sum(1), draws.n, draws.beta, provenance=draws.provenance
    )
