"""Choosing the polynomial degree with AIC, BIC, HQIC (C-30 - C-32).

MLE is closed form: l = -n/2 (log(2 pi RSS / n) + 1), k = degree + 2.
"""

import math

import torch

from dl_complexity.criteria import AIC, BIC, HQIC, FitStatistics

DT = torch.float64
TRUE_COEF = (0.3, -0.8, 0.6)  # true degree 2
MAX_DEGREE = 5


def profile_log_likelihoods(seeds, n, max_degree, coef, *, seed):
    """Max log-likelihood for every degree, [seeds, max_degree + 1]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(seeds, n, generator=g, dtype=DT) * 2 - 1
    y = sum(c * x**j for j, c in enumerate(coef))
    y = y + torch.randn(seeds, n, generator=g, dtype=DT)
    design = torch.stack([x**j for j in range(max_degree + 1)], -1)
    columns = []
    for degree in range(max_degree + 1):
        a = design[..., : degree + 1]
        fit = a @ torch.linalg.lstsq(a, y[..., None]).solution
        rss = (y - fit.squeeze(-1)).square().sum(-1)
        columns.append(-0.5 * n * (torch.log(2 * math.pi * rss / n) + 1))
    return torch.stack(columns, -1)


def selected_degrees(criterion, log_likelihoods, n):
    """Degree with the smallest criterion for each dataset."""
    choices = []
    for row in log_likelihoods:
        values = [
            float(
                criterion.estimate(
                    inputs=FitStatistics(ll, k=degree + 2, n=n)
                ).primary.value
            )
            for degree, ll in enumerate(row)
        ]
        choices.append(min(range(len(values)), key=values.__getitem__))
    return torch.tensor(choices)


def test_bic_is_consistent_and_aic_is_not():
    """BIC finds the true degree more often as n grows, AIC does not.

    Limits with 3 extra models (chi2 simulation): BIC 0.998 and HQIC 0.958
    at n = 1e4, AIC 0.760 for any n.
    """
    fractions = {name: [] for name in ("AIC", "BIC", "HQIC")}
    overfit = {}
    for n in (100, 1000, 10_000):
        ll = profile_log_likelihoods(200, n, MAX_DEGREE, TRUE_COEF, seed=n)
        for criterion in (AIC(), BIC(), HQIC()):
            name = type(criterion).__name__
            chosen = selected_degrees(criterion, ll, n)
            fractions[name].append(float((chosen == 2).double().mean()))
            overfit[name, n] = float((chosen > 2).double().mean())
    bic = fractions["BIC"]
    assert bic[0] < bic[1] < bic[2] or bic[1] == bic[2] == 1.0
    assert bic[1] >= 0.9 and bic[2] >= 0.95
    assert overfit["BIC", 10_000] <= 0.02
    # small n: BIC underfits more than AIC
    assert bic[0] < fractions["AIC"][0]
    # AIC keeps overfitting (limit 0.24)
    assert 0.12 <= overfit["AIC", 10_000] <= 0.36
    assert 0.12 <= overfit["AIC", 1000] <= 0.36
    # HQIC is consistent too, but slower than BIC
    assert fractions["HQIC"][2] >= 0.9
    assert overfit["HQIC", 10_000] >= overfit["BIC", 10_000]


def test_one_superfluous_parameter_overfit_probabilities():
    """One extra parameter: P(pick the bigger model) = P(chi2_1 > penalty).

    n = 1000: AIC 0.1573, HQIC 0.0493, BIC 0.0086; tolerance 4 binomial SE.
    """
    n, seeds = 1000, 2000
    ll = profile_log_likelihoods(seeds, n, 1, (0.5,), seed=12345)  # true degree 0
    references = {
        AIC: 0.15730,
        HQIC: 0.04929,
        BIC: 0.00858,
    }
    for method, p in references.items():
        chosen = selected_degrees(method(), ll, n)
        rate = float((chosen == 1).double().mean())
        tolerance = 4 * math.sqrt(p * (1 - p) / seeds)
        assert abs(rate - p) <= tolerance, (method.__name__, rate, p)
