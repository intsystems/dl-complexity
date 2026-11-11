"""Exact answers for the sampler tests, no sampling.

LinearGaussian: y = X w + eps, the tempered posterior is Gaussian
(regular, lambda = d/2).
ProductModel: y = a b x + eps, b | a is Gaussian, 1D quadrature over a
(singular at ab = 0: lambda = 1/2 instead of d/2 = 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

LOG_2PI = math.log(2 * math.pi)


def make_regression_data(n, weights, *, sigma, seed, dtype=torch.float64):
    """Draw x ~ N(0, I) and y = x @ weights + sigma * eps with a private generator.

    The private generator keeps the dataset independent of the global RNG and of
    any random numbers consumed by the sampler under test.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    x = torch.randn(n, weights.numel(), generator=generator, dtype=torch.float64)
    noise = torch.randn(n, generator=generator, dtype=torch.float64)
    y = x @ weights + sigma * noise
    return x.to(dtype), y[:, None].to(dtype)


def _pointwise_gaussian_waic_terms(residual, pred_var, sigma2):
    """Exact per-observation lppd and Var(log p) for a Gaussian-mean posterior.

    With prediction mean mu, predictive-mean variance s^2 and r = y - mu:
    E[p(y|w)] = N(y | mu, sigma^2 + s^2) and, writing l = c - (r - z)^2 / (2 sigma^2)
    with z ~ N(0, s^2), Var(l) = (r^2 s^2 + s^4 / 2) / sigma^4.
    """
    lppd = -0.5 * (LOG_2PI + torch.log(sigma2 + pred_var)) - 0.5 * residual**2 / (
        sigma2 + pred_var
    )
    variance = (residual**2 * pred_var + 0.5 * pred_var**2) / sigma2**2
    return lppd, variance


@dataclass(frozen=True)
class LinearGaussian:
    """Closed-form tempered posterior of Bayesian linear regression with known noise."""

    x: torch.Tensor  # [n, d]
    y: torch.Tensor  # [n, 1]
    sigma: float
    alpha: float

    def __post_init__(self):
        object.__setattr__(self, "x", self.x.double())
        object.__setattr__(self, "y", self.y.double().reshape(-1, 1))

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def d(self):
        return self.x.shape[1]

    def posterior(self, beta=1.0, gamma=0.0, w0=None):
        """Return mean [d] and covariance [d, d] of the tempered, localized target."""
        sigma2 = self.sigma**2
        eye = torch.eye(self.d, dtype=torch.float64)
        w0 = torch.zeros(self.d, dtype=torch.float64) if w0 is None else w0.double()
        precision = (self.alpha + gamma) * eye + beta * self.x.T @ self.x / sigma2
        rhs = beta * (self.x.T @ self.y).reshape(-1) / sigma2 + gamma * w0
        covariance = torch.linalg.inv(precision)
        covariance = 0.5 * (covariance + covariance.T)
        return covariance @ rhs, covariance

    def total_nll(self, w):
        """Total NLL nL_n(w) for a weight vector [d]."""
        residual = self.y.reshape(-1) - self.x @ w.double()
        return (
            0.5 * self.n * (LOG_2PI + math.log(self.sigma**2))
            + 0.5 * float(residual @ residual) / self.sigma**2
        )

    def mle(self):
        return torch.linalg.lstsq(self.x, self.y).solution.reshape(-1)

    def expected_total_nll(self, beta=1.0, gamma=0.0, w0=None):
        """E[nL_n(w)] under the tempered target: WBIC when beta = 1 / log n."""
        mean, covariance = self.posterior(beta, gamma, w0)
        trace = float(torch.trace(self.x.T @ self.x @ covariance))
        return self.total_nll(mean) + 0.5 * trace / self.sigma**2

    def waic(self):
        """Population WAIC (S -> infinity) under the ordinary posterior, beta = 1."""
        mean, covariance = self.posterior(1.0)
        residual = self.y.reshape(-1) - self.x @ mean
        pred_var = ((self.x @ covariance) * self.x).sum(1)
        lppd, variance = _pointwise_gaussian_waic_terms(
            residual, pred_var, torch.tensor(self.sigma**2, dtype=torch.float64)
        )
        return {
            "lppd": float(lppd.sum()),
            "p_waic": float(variance.sum()),
            "waic": float(-2 * (lppd.sum() - variance.sum())),
        }

    def learning_coefficient(self, beta=None):
        """Finite-n lambda_hat = (E_beta[nL_n] - nL_n(w_mle)) / log n, exact."""
        beta = 1 / math.log(self.n) if beta is None else beta
        excess = self.expected_total_nll(beta) - self.total_nll(self.mle())
        return excess * beta


@dataclass(frozen=True)
class ProductModel:
    """y = a b x + eps with iid N(0, 1/alpha) priors on a and b; 1D quadrature in a."""

    x: torch.Tensor  # [n]
    y: torch.Tensor  # [n]
    sigma: float
    alpha: float
    grid_size: int = 400_001
    grid_halfwidth_sd: float = 12.0

    def __post_init__(self):
        object.__setattr__(self, "x", self.x.double().reshape(-1))
        object.__setattr__(self, "y", self.y.double().reshape(-1))

    @property
    def n(self):
        return self.x.numel()

    def _grid(self, beta):
        """Normalised quadrature weights over a and the Gaussian b | a parameters."""
        sigma2 = self.sigma**2
        sxx = float(self.x @ self.x)
        sxy = float(self.x @ self.y)
        half = self.grid_halfwidth_sd / math.sqrt(self.alpha)
        a = torch.linspace(-half, half, self.grid_size, dtype=torch.float64)
        precision = self.alpha + beta * a**2 * sxx / sigma2
        mean = beta * a * sxy / (sigma2 * precision)
        # log of prior(a) * integral over b, up to a constant
        log_weight = (
            -0.5 * self.alpha * a**2
            + 0.5 * torch.log(self.alpha / precision)
            + 0.5 * precision * mean**2
        )
        weight = torch.exp(log_weight - log_weight.max())
        # trapezoid rule, the integrand vanishes at the grid ends
        weight[0] *= 0.5
        weight[-1] *= 0.5
        weight = weight / weight.sum()
        return a, weight, mean, 1 / precision

    def moments(self, beta=1.0):
        """E[a^2], E[b^2], E[ab] and E[(ab)^2] under the tempered posterior."""
        a, weight, mean_b, var_b = self._grid(beta)
        second_b = mean_b**2 + var_b
        return {
            "E[a^2]": float(weight @ a**2),
            "E[b^2]": float(weight @ second_b),
            "E[ab]": float(weight @ (a * mean_b)),
            "E[(ab)^2]": float(weight @ (a**2 * second_b)),
        }

    def total_nll(self, a, b):
        residual = self.y - a * b * self.x
        return (
            0.5 * self.n * (LOG_2PI + math.log(self.sigma**2))
            + 0.5 * float(residual @ residual) / self.sigma**2
        )

    def min_total_nll(self):
        """nL_n at the MLE of the product c = ab, which is attained by a = b = sqrt|c|."""
        c = float(self.x @ self.y) / float(self.x @ self.x)
        return self.total_nll(math.copysign(math.sqrt(abs(c)), c), math.sqrt(abs(c)))

    def expected_total_nll(self, beta=1.0):
        a, weight, mean_b, var_b = self._grid(beta)
        sxx, sxy, syy = (
            float(self.x @ self.x),
            float(self.x @ self.y),
            float(self.y @ self.y),
        )
        quadratic = syy - 2 * a * mean_b * sxy + a**2 * (mean_b**2 + var_b) * sxx
        constant = 0.5 * self.n * (LOG_2PI + math.log(self.sigma**2))
        return constant + 0.5 * float(weight @ quadratic) / self.sigma**2

    def waic(self, max_entries=10_000_000):
        """Population WAIC under the ordinary posterior by mixing b | a over the a-grid."""
        a, weight, mean_b, var_b = self._grid(1.0)
        keep = weight > 1e-300
        a, weight, mean_b, var_b = a[keep], weight[keep], mean_b[keep], var_b[keep]
        chunk = max(1, max_entries // a.numel())  # bounds memory at O(max_entries)
        sigma2 = self.sigma**2
        c = -0.5 * (LOG_2PI + math.log(sigma2))
        lppd, variance = [], []
        for start in range(0, self.n, chunk):
            x = self.x[start : start + chunk]
            y = self.y[start : start + chunk]
            mu = (a * mean_b)[:, None] * x[None, :]
            s2 = (a**2 * var_b)[:, None] * x[None, :] ** 2
            r = y[None, :] - mu
            log_pred = -0.5 * (LOG_2PI + torch.log(sigma2 + s2)) - 0.5 * r**2 / (
                sigma2 + s2
            )
            lppd.append(torch.logsumexp(log_pred + torch.log(weight)[:, None], dim=0))
            q1 = r**2 + s2
            q2 = r**4 + 6 * r**2 * s2 + 3 * s2**2
            first = weight @ (c - q1 / (2 * sigma2))
            second = weight @ (c**2 - c * q1 / sigma2 + q2 / (4 * sigma2**2))
            variance.append(second - first**2)
        lppd, variance = torch.cat(lppd), torch.cat(variance)
        return {
            "lppd": float(lppd.sum()),
            "p_waic": float(variance.sum()),
            "waic": float(-2 * (lppd.sum() - variance.sum())),
        }

    def learning_coefficient(self, beta=None):
        """Finite-n lambda_hat = (E_beta[nL_n] - min nL_n) * beta, exact up to quadrature."""
        beta = 1 / math.log(self.n) if beta is None else beta
        return (self.expected_total_nll(beta) - self.min_total_nll()) * beta
