"""Reproduce the analytic examples in main.tex; no training data or network needed.

Run: python theory/reproduce.py (from the repository root).
All generated files are relative to this script, independent of the working directory.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from scipy.stats import binom

ROOT = Path(__file__).resolve().parent
LN2 = math.log(2)
BETA = 1 + 1 / (math.e * LN2)
KAPPA = 1 + BETA + 2 * math.log2(1 + BETA)


def delta_length(j: int) -> int:
    """Exact integer arithmetic, including j=1."""
    k = j.bit_length()
    return k + 2 * (k.bit_length() - 1)


def geometric_length(h: float | np.ndarray) -> float | np.ndarray:
    """Elias delta mean for Geom(exp(-h)), via dyadic survival probabilities.

    Stop after p * 2**j >= 50 for the smallest p in the array. The omitted
    tail is at most 3*exp(p)*exp(-p*2**(j+1))/(1-exp(-p*2**(j+1))) < 1e-42.
    This is a truncation bound, not a bound on floating point rounding.
    """
    h_arr = np.asarray(h, dtype=float)
    if np.any(~np.isfinite(h_arr)) or np.any(h_arr < 0) or np.any(h_arr > 650):
        raise ValueError("h must be finite and between 0 and 650")
    p = np.exp(-h_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_survival = np.where(h_arr <= LN2, np.log(-np.expm1(-h_arr)), np.log1p(-p))
    result = np.ones_like(h_arr)
    last = max(1, math.ceil((float(np.max(h_arr)) + math.log(50)) / LN2))
    for j in range(1, last + 1):
        jump = 3 if ((j + 1) & j) == 0 else 1
        result += jump * np.exp((2.0**j - 1) * log_survival)
    return float(result) if result.ndim == 0 else result


def upper_cost(h):
    d = np.asarray(h) / LN2
    return d + 2 * np.log2(1 + d) + KAPPA


def coefficient(a: float, max_mean: float | None = None):
    """Grid localization and bounded refinement; numerical, not interval proof."""
    hi = 40.0 if max_mean is None else math.log(max_mean)
    grid = np.geomspace(1e-6, hi, 1600)
    values = geometric_length(grid) / np.log1p(grid / a)
    candidates = []
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            opt = minimize_scalar(
                lambda h: geometric_length(h) / math.log1p(h / a),
                bounds=(grid[i - 1], grid[i + 1]), method="bounded",
                options={"xatol": 1e-12},
            )
            candidates.append((float(opt.fun), float(opt.x)))
    candidates.append((float(values[-1]), hi))
    value, h = min(candidates)
    # Outside [1e-6, 40], elementary lower bounds exclude a smaller minimum.
    if max_mean is None:
        assert 1 / math.log1p(1e-6 / a) > value
        assert hi / (LN2 * math.log1p(hi / a)) > value
    return {"a": a, "bits_per_ln": value, "h": h, "r": 1 + h / a,
            "mean_candidates": math.exp(h)}


def deterministic_costs(n_max: int, a=0.25, b=1.0, max_mean=None):
    costs = np.full((n_max + 1, n_max + 1), np.inf)
    for n in range(n_max):
        sizes = np.arange(1, n_max - n + 1)
        h = sizes * math.log1p(a / (n + b))
        vals = geometric_length(h)
        if max_mean is not None:
            vals = np.where(h <= math.log(max_mean) + 1e-14, vals, np.inf)
        costs[n, n + sizes] = vals
    return costs


def shortest_schedule(costs, n_max):
    value = np.full(n_max + 1, np.inf)
    previous = np.full(n_max + 1, -1, dtype=int)
    value[0] = 0
    for end in range(1, n_max + 1):
        choices = value[:end] + costs[:end, end]
        previous[end] = int(np.argmin(choices))
        value[end] = choices[previous[end]]
    return value, previous


def schedule(previous, end):
    batches = []
    while end:
        start = int(previous[end])
        batches.append(end - start)
        end = start
    return batches[::-1]


def path_cost(costs, batches):
    value, n = 0.0, 0
    for size in batches:
        value += float(costs[n, n + size])
        n += size
    return value


def all_schedules(n):
    for mask in range(1 << (n - 1)):
        boundaries = [0] + [i for i in range(1, n) if mask & (1 << (i - 1))] + [n]
        yield [b - a for a, b in zip(boundaries, boundaries[1:])]


def noisy_costs(n_max, q=0.2, a=1.0, b=1.0):
    """Public deterministic schedules: average over Bin(n,q), then Bin(B,q).

    Qiu et al., Appendix B.2, eqs. (34)--(38), reduced by sufficient counts.
    No Monte Carlo and no binomial tail truncation are used.
    """
    exact = np.full((n_max + 1, n_max + 1), np.inf)
    bound = np.full_like(exact, np.inf)
    for n in range(n_max):
        s = np.arange(n + 1)
        history_prob = binom.pmf(s, n, q)
        p = ((s + a) / (n + a + b))[:, None]
        per_sample_kl = q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
        for size in range(1, n_max - n + 1):
            k = np.arange(size + 1)[None, :]
            log_ratio = k * np.log(q / p) + (size - k) * np.log((1 - q) / (1 - p))
            ratio = np.exp(log_ratio)
            lower_p = np.where(q > p, binom.cdf(k - 1, size, p), binom.sf(k, size, p))
            lower_q = np.where(q > p, binom.cdf(k - 1, size, q), binom.sf(k, size, q))
            mean = np.maximum(1.0, 1 + ratio * lower_p - lower_q)
            chosen_prob = binom.pmf(k, size, q)
            cost_by_history = np.sum(chosen_prob * geometric_length(np.log(mean)), axis=1)
            exact[n, n + size] = history_prob @ cost_by_history
            bound[n, n + size] = history_prob @ upper_cost(size * per_sample_kl[:, 0])
    return exact, bound


def verify_series():
    errors = []
    for h in [0, 0.05, 0.3, 0.8, 1.5, 2.0]:
        p = math.exp(-h)
        stop = 1 if p == 1 else math.ceil(math.log(1e-15) / math.log1p(-p))
        direct = sum(delta_length(j) * p * (1 - p)**(j - 1) for j in range(1, stop + 1))
        errors.append(abs(direct - geometric_length(h)))
    assert max(errors) < 2e-12
    # Rare proposals need log1p(-exp(-h)), not log(-expm1(-h)).
    for h in [40.0, 100.0, 500.0]:
        val = geometric_length(h)
        assert h / LN2 <= val <= h / LN2 + 2 * math.log2(1 + h / LN2) + 1
    return max(errors)


def verify_noisy_transition():
    """Enumerate every binary string, independent of the binomial-CDF reduction."""
    q, p, size = 0.2, 0.43, 3
    strings = list(itertools.product([0, 1], repeat=size))
    proposal = np.array([p**sum(x) * (1 - p)**(size - sum(x)) for x in strings])
    target = np.array([q**sum(x) * (1 - q)**(size - sum(x)) for x in strings])
    ratios = target / proposal
    means = 1 + np.sum(proposal[None, :] * np.maximum(ratios[:, None] - ratios[None, :], 0), axis=1)
    direct = float(target @ geometric_length(np.log(means)))
    k = np.arange(size + 1)
    r = (q / p)**k * ((1 - q) / (1 - p))**(size - k)
    reduced = float(binom.pmf(k, size, q) @ geometric_length(np.log(np.maximum(1, 1 + r * binom.sf(k, size, p) - binom.sf(k, size, q)))))
    assert abs(direct - reduced) < 1e-12
    return abs(direct - reduced)


def dynamics(t_max=2**18):
    records = []
    samples = {}
    for a0, b0 in [(0.1, 0.1), (0.2, 0.05), (1.0, 0.05)]:
        a, b, total = a0, b0, 0.0
        points = []
        for t in range(t_max + 1):
            risk = a * b / 2
            if t >= 32 and (t & (t - 1)) == 0:
                points.append([t, risk, total, abs(a - b)])
            if t == t_max:
                break
            total += risk
            eta = 1 / (t + 10)
            a, b = a * (1 - eta * b)**2, b * (1 - eta * a)**2
        imbalance = abs(a - b)
        tail_bound = imbalance * a * b / (t_max + 9)
        if a0 != b0:
            observed = -math.log2(points[-1][1] / points[-2][1])
            assert abs(observed - 2 * imbalance) < 0.035
        records.append({"a0": a0, "b0": b0, "T": t_max, "risk": risk,
                        "S": total, "D_upper": imbalance,
                        "D_lower": imbalance - tail_bound,
                        "alpha_upper": 2 * imbalance,
                        "tail_bound": tail_bound})
        samples[f"{a0},{b0}"] = points
    # Balanced recurrence: reciprocal norm minus twice harmonic time stays bounded.
    s, harmonic = 0.1, 0.0
    residuals = []
    for t in range(t_max):
        eta = 1 / (t + 10)
        s *= (1 - eta * s)**2
        harmonic += eta
        if t > t_max // 2:
            residuals.append(1 / s - 2 * harmonic)
    assert max(residuals) - min(residuals) < 1e-5
    return records, samples


def main():
    (ROOT / "generated").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    checks = {"series_direct_sum_error": verify_series(),
              "pfr_binomial_reduction_error": verify_noisy_transition()}
    coefs = [coefficient(a) for a in [0.1, 0.25, 0.5, 1.0]]
    constrained = [dict(coefficient(0.25, m), M=m) for m in [1.25, 1.5, 2, 3, 4]]
    # Check the constrained theorem against finite shortest paths, not only a
    # numerical minimization of its asymptotic expression.
    constrained_finite = []
    for item in constrained:
        c_m = deterministic_costs(256, max_mean=item["M"])
        v_m, pred_m = shortest_schedule(c_m, 256)
        batches_m = schedule(pred_m, 256)
        n = 0
        largest_mean = 0.0
        for batch in batches_m:
            largest_mean = max(largest_mean, math.exp(batch * math.log1p(0.25 / (n + 1))))
            n += batch
        lower_m = item["bits_per_ln"] * math.log((256 + 1.25) / 1.25)
        assert largest_mean <= item["M"] + 1e-12
        assert v_m[256] >= lower_m - 1e-9
        constrained_finite.append({"M": item["M"], "N": 256,
                                   "bits": float(v_m[256]), "lower_bound": lower_m,
                                   "largest_mean": largest_mean, "batches": batches_m})
    costs = deterministic_costs(512)
    values, previous = shortest_schedule(costs, 512)
    brute = min((path_cost(costs, x), x) for x in all_schedules(8))
    assert abs(brute[0] - values[8]) < 1e-12
    checks["dp_exhaustive_N8_error"] = abs(brute[0] - values[8])
    schedules = [{"batches": x, "bits": path_cost(costs, x)}
                 for x in [schedule(previous, 63), [7, 56], [63], [1] * 63]]
    assert schedule(previous, 63) == [3, 12, 48]
    assert abs(values[63] - 8.071) < 0.001
    assert abs(coefs[1]["bits_per_ln"] - 1.994) < 0.001
    boundary_upper = np.full((64, 64), np.inf)
    for n in range(63):
        sizes = np.arange(1, 64 - n)
        boundary_upper[n, n + sizes] = upper_cost(sizes * math.log1p(0.25 / (n + 1)))
    _, upper_previous = shortest_schedule(boundary_upper, 63)
    assert schedule(upper_previous, 63) == [7, 56]
    noise, upper = noisy_costs(62)
    nv, npv = shortest_schedule(noise, 62)
    uv, upv = shortest_schedule(upper, 62)
    ns, us = schedule(npv, 62), schedule(upv, 62)
    assert ns == [3, 15, 44] and us == [7, 55]
    noisy = {"q": 0.2, "a": 1, "b": 1, "N": 62, "exact_schedule": ns,
             "exact_bits": float(nv[62]), "bound_schedule": us,
             "bound_schedule_actual_bits": path_cost(noise, us),
             "bound_at_exact_schedule": path_cost(upper, ns),
             "bound_at_bound_schedule": float(uv[62])}
    s_final = np.arange(63)
    p_final = (s_final + 1) / 64
    noisy["terminal_expected_KL"] = float(binom.pmf(s_final, 62, 0.2) @ (
        0.2 * np.log(0.2 / p_final) + 0.8 * np.log(0.8 / (1 - p_final))))
    noisy["saving_percent"] = 100 * (1 - noisy["exact_bits"] / noisy["bound_schedule_actual_bits"])
    dyn, samples = dynamics()
    # Finite-N check of the new theorem, for different prior second parameters.
    general = []
    for b in [0.5, 1, 2, 4]:
        c = deterministic_costs(256, b=b)
        v, prev = shortest_schedule(c, 256)
        lower = coefs[1]["bits_per_ln"] * math.log((256 + b + 0.25) / (b + 0.25))
        assert v[256] >= lower - 1e-9
        general.append({"a": 0.25, "b": b, "N": 256, "bits": float(v[256]),
                        "lower_bound": lower, "batches": schedule(prev, 256)})
    # Pinsker/chi-square endpoint bound, checked over all binary suffixes h=6.
    max_kl = 0.0
    for labels in itertools.product([0, 1], repeat=6):
        def advance(p):
            for y in labels:
                p = (p + 1 / 8 + 3 * y / 4) / 2
            return p
        approx = advance(0.5)
        for old in [1 / 8, 7 / 8]:
            target = advance(old)
            kl = target * math.log(target / approx) + (1 - target) * math.log((1 - target) / (1 - approx))
            max_kl = max(max_kl, kl)
    assert max_kl <= (9 / 7) * 4**-6 < 1e-3
    checks["suffix_h6_max_endpoint_KL"] = max_kl
    # Joint zero-block probability equals the product of sequential one-step
    # probabilities. This compares two independent analytic expressions.
    joint_errors = []
    for a, b, n, size in [(0.25, 1, 0, 63), (0.5, 2, 7, 19), (1, 0.5, 0, 8)]:
        sequential_h = sum(math.log1p(a / (b + n + j)) for j in range(size))
        joint_h = betaln(a, b + n) - betaln(a, b + n + size)
        joint_errors.append(abs(sequential_h - joint_h))
        assert geometric_length(joint_h) >= joint_h / LN2
        assert joint_h <= size * math.log1p(a / (b + n)) + 1e-12
    checks["joint_beta_vs_sequential_error"] = max(joint_errors)
    assert max(joint_errors) < 1e-12
    output = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
              "checks": checks, "coefficient": coefs, "constrained": constrained,
              "constrained_finite": constrained_finite,
              "deterministic_N63": schedules, "noisy_N62": noisy,
              "general_beta": general, "gradient_dynamics": dyn,
              "kappa": KAPPA}
    (ROOT / "generated/results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [r"\begin{tabular}{rrr}", r"\toprule", r"$a$ & $r_*$ & $A(a)$, бит/$\ln(1/\varepsilon)$\\", r"\midrule"]
    rows += [f"{x['a']:.2f} & {x['r']:.3f} & {x['bits_per_ln']:.6f}\\\\" for x in coefs]
    rows += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "generated/coefficient_table.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    r = np.linspace(1.04, 12, 500)
    ax[0].plot(r, geometric_length(0.25 * (r - 1)) / np.log(r), label="Замороженный предиктор")
    ax[0].axhline(0.25 / LN2, color="0.4", linestyle="--", label="Совместный: предельный коэффициент")
    ax[0].scatter([coefs[1]["r"]], [coefs[1]["bits_per_ln"]], color="C1", zorder=3)
    ax[0].set(xlabel="Отношение объёмов выборки r", ylabel=r"Бит на единицу $\ln(1/\varepsilon)$", ylim=(0, 8))
    ax[0].legend(fontsize=8)
    ns_grid = np.arange(1, 513)
    joint = geometric_length(betaln(0.25, 1) - np.array([betaln(0.25, 1 + n) for n in ns_grid]))
    ax[1].plot(ns_grid, values[1:], label="Замороженный: оптимальное расписание")
    ax[1].plot(ns_grid, joint, label="Совместный: один пакет")
    ax[1].set(xlabel="Конечное число наблюдений N", ylabel="Ожидаемая длина, бит")
    ax[1].legend(fontsize=8)
    fig.savefig(ROOT / "figures/protocol_costs.pdf")
    fig.savefig(ROOT / "figures/protocol_costs.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"checks": checks, "a_quarter": coefs[1], "N63": schedules[:3],
                      "noisy": noisy, "general_beta": general, "dynamics": dyn}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
