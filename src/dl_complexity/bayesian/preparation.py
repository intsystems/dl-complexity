"""Explicit Bayesian preparation, never invoked implicitly by estimate/check."""

from collections.abc import Callable
import math

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, kl_divergence

from dl_complexity._validation import positive_int, scalar
from dl_complexity.bayesian.inputs import KFACFactors, KroneckerBlock, VariationalTerms
from dl_complexity.errors import InputError
from dl_complexity.results import Provenance


def prepare_variational_terms(
    q: Distribution,
    prior: Distribution,
    total_nll: Callable[[Tensor], Tensor],
    *,
    n_samples: int = 1,
    n: int | None = None,
    analytic_kl: bool = True,
    provenance: Provenance | None = None,
) -> VariationalTerms:
    r"""Explicitly sample a supplied q and prepare total negative-ELBO terms.

    Args:
        q (Distribution): Already specified variational distribution. Uses
            rsample when available, otherwise sample without a score-function
            gradient estimator.
        prior (Distribution): Normalised prior on the same joint parameter law.
        total_nll (Callable): Function theta -> scalar total-data negative
            log-likelihood, not mean minibatch loss or regularised objective.
        n_samples (int): Positive number S of parameter draws.
        n (int, optional): Observation count recorded without implicit scaling.
        analytic_kl (bool): Request registered torch.distributions KL; otherwise
            estimate joint log q(theta) - log prior(theta) using the same draws.
        provenance (Provenance, optional): Declared artifact identities.

    Returns:
        VariationalTerms: Monte Carlo expected NLL and analytic/MC KL in nats.

    Raises:
        InputError: If counts/NLL scalars are invalid or requested analytic KL
            is not registered. Distribution/support errors can also propagate.

    Notes:
        No fitting is performed. This explicit preparation consumes RNG. Each
        log_prob is summed to its OWN joint scalar before subtraction, avoiding
        broadcasting errors between batch-Normal and Independent representations.
        Work is sampling plus S total_nll calls and KL/density evaluation;
        storage includes all S draws and their retained autograd graphs.
    """
    positive_int(n_samples, "n_samples")
    if n is not None:
        positive_int(n, "n")
    samples = q.rsample((n_samples,)) if q.has_rsample else q.sample((n_samples,))
    nll = torch.stack(
        [scalar(total_nll(theta), "total_nll(theta)") for theta in samples]
    ).mean()
    if analytic_kl:
        try:
            kl = kl_divergence(q, prior).sum()
        except NotImplementedError as error:
            raise InputError(
                "No analytic KL registered; explicitly request analytic_kl=False for MC"
            ) from error
        estimator = "analytic"
    else:
        # Reduce each representation to its own joint log-density BEFORE
        # subtraction: batch Normal and event Independent(Normal) can represent
        # the same joint law, but their raw log_prob shapes differ.
        kl = torch.stack(
            [q.log_prob(theta).sum() - prior.log_prob(theta).sum() for theta in samples]
        ).mean()
        estimator = "monte_carlo"
    return VariationalTerms(nll, kl, n=n, kl_estimator=estimator, provenance=provenance)


def prepare_gaussian_linear_kfac(
    model: nn.Linear,
    x: Tensor,
    y: Tensor,
    *,
    noise_std: float,
    prior_precision: float,
    provenance: Provenance | None = None,
) -> KFACFactors:
    r"""Prepare exact Gaussian-regression curvature for one plain Linear layer.

    Args:
        model (nn.Linear): Exact built-in Linear type, with optional bias;
            all its weights/biases are selected. Must already be MAP if the
            subsequent Laplace value is to represent evidence.
        x (Tensor): Finite inputs [N, input_features], N >= 1.
        y (Tensor): Finite targets [N, output_features], matching model dtype/device.
        noise_std (float): Positive known observation standard deviation sigma.
        prior_precision (float): Positive alpha for independent N(0, alpha^-1)
            weights/biases; normalising constants are included.
        provenance (Provenance, optional): Declared artifact identities.

    Returns:
        KFACFactors: One block A=X_aug.T@X_aug/N, G=I/sigma^2, scale=N;
        total likelihood and normalised prior at the supplied weights.

    Raises:
        InputError: If model type, shape, dtype/device, finiteness or scales fail.

    Notes:
        X_aug appends a column of ones for bias. With p augmented features and
        g outputs, work is O(N*p*g + N*p^2 + g^2), additional numeric storage
        O(N*p + N*g + p^2 + g^2 + p*g), plus autograd. The function directly
        performs the linear algebra, does not call an arbitrary model forward,
        fit parameters, build a general-network collector, or record telemetry.
    """
    if type(model) is not nn.Linear:
        raise InputError("This exact preparer supports a single plain nn.Linear only")
    if (
        not math.isfinite(noise_std)
        or noise_std <= 0
        or not math.isfinite(prior_precision)
        or prior_precision <= 0
    ):
        raise InputError("noise_std and prior_precision must be finite and positive")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != model.in_features:
        raise InputError("x must have shape [n >= 1, in_features]")
    if y.shape != (x.shape[0], model.out_features):
        raise InputError("y must have shape [n, out_features]")
    if (
        x.device != model.weight.device
        or y.device != x.device
        or x.dtype != model.weight.dtype
        or y.dtype != x.dtype
    ):
        raise InputError("x, y and Linear parameters must have matching dtype/device")
    if (
        not x.is_floating_point()
        or not bool(torch.isfinite(x).all())
        or not bool(torch.isfinite(y).all())
    ):
        raise InputError("x/y must contain finite floating values")
    prediction = x @ model.weight.mT
    selected = [model.weight.reshape(-1)]
    if model.bias is not None:
        prediction = prediction + model.bias
        augmented = torch.cat([x, x.new_ones((x.shape[0], 1))], dim=1)
        selected.append(model.bias.reshape(-1))
    else:
        augmented = x
    parameters = torch.cat(selected)
    log_likelihood = torch.distributions.Normal(prediction, noise_std).log_prob(y).sum()
    log_prior = (
        0.5 * parameters.numel() * (math.log(prior_precision) - math.log(2 * math.pi))
        - 0.5 * prior_precision * parameters.square().sum()
    )
    a = augmented.mT @ augmented / x.shape[0]
    g = torch.eye(model.out_features, dtype=x.dtype, device=x.device) / noise_std**2
    return KFACFactors(
        log_likelihood,
        log_prior,
        (KroneckerBlock(a, g, float(x.shape[0])),),
        prior_precision,
        curvature_kind="exact_gaussian_linear_hessian",
        provenance=provenance,
    )
