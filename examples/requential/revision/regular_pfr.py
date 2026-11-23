"""Expected Elias lengths for fixed batch schedules at matched terminal risk."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

from common import LN2, bern_kl, bound, save
from quartic_pfr_cost import message_bits


def log_index_mean(batch, s, p, q):
    """E[J|S=s] from the conditional geometric PFR law, returned in log space."""
    log_ratio = s * (np.log(q) - np.log(p)) + (batch - s) * (np.log1p(-q) - np.log1p(-p))
    # Split on outcomes with a strictly smaller likelihood ratio than the target.
    increasing = q >= p
    first = log_ratio + np.where(increasing, binom.logcdf(s - 1, batch, p),
                                 binom.logsf(s, batch, p))
    second = np.where(increasing, binom.logsf(s - 1, batch, q), binom.logcdf(s, batch, q))
    return np.where(p == q, 0.0, np.maximum(0.0, np.logaddexp(first, second)))


def problem(q):
    @lru_cache(None)
    def law(n):
        # At most 2e-14 omitted mass per marginal distribution.
        lo, hi = binom.ppf([1e-14, 1 - 1e-14], n, q).astype(int)
        k = np.arange(lo, hi + 1)
        weights = binom.pmf(k, n, q)
        return k, weights / weights.sum()

    @lru_cache(None)
    def risk(n):
        k, w = law(n)
        return float(w @ bern_kl(q, (k + 1) / (n + 2)))

    def costs(n, batch):
        k, w = law(n)
        s, v = law(batch)
        p = (k + 1) / (n + 2)
        h = log_index_mean(batch, s[None, :], p[:, None], q)
        actual = w @ (message_bits(h) @ v)
        rhs = w @ bound(batch * bern_kl(q, p) / LN2)
        return np.array([actual, rhs])

    return risk, costs


def optimizers(horizon, costs):
    best = np.full((horizon + 1, 2), np.inf)
    best[0] = 0
    previous = np.zeros((horizon + 1, 2), dtype=int)
    for end in range(1, horizon + 1):
        for n in range(end):
            candidate = best[n] + costs(n, end - n)
            improved = candidate < best[end]
            best[end, improved] = candidate[improved]
            previous[end, improved] = n
    return previous


def recover(previous, n, column):
    sizes = []
    while n:
        start = int(previous[n, column])
        sizes.append(n - start)
        n = start
    return sizes[::-1]


def geometric(n, growth):
    sizes = []
    total = 0
    while total < n:
        b = min(n - total, max(1, round((growth - 1) * total)))
        sizes.append(b)
        total += b
    return sizes


def compare(q, epsilons, cap):
    risk, costs = problem(q)
    targets = {}
    for eps in epsilons:
        target = next((n for n in range(cap + 1) if risk(n) <= eps), None)
        if target is None:
            raise ValueError(f"No N <= {cap} reaches epsilon={eps} for q={q}; increase --cap.")
        targets[eps] = target
    previous = optimizers(max(targets.values()), costs)
    rows = []
    for eps, n in targets.items():
        schedules = {"pfr_dp": recover(previous, n, 0), "bound_dp": recover(previous, n, 1),
                     "unit": [1] * n, "single": [n] if n else []}
        for r in (1.5, 2., 4., 8.):
            schedules[f"geom{r:g}"] = geometric(n, r)
        for name, sizes in schedules.items():
            start = 0
            total = np.zeros(2)
            for batch in sizes:
                total += costs(start, batch)
                start += batch
            if start != n or total[0] > total[1] + 1e-8:
                raise ArithmeticError("Schedule or code-bound check failed")
            rows.append(dict(q=q, epsilon_nats=eps, n=n, final_kl_nats=risk(n),
                             policy=name, batches=json.dumps(sizes),
                             expected_pfr_bits=total[0], upper_bound_bits=total[1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", nargs="+", type=float, default=[.37, .2])
    parser.add_argument("--epsilon", nargs="+", type=float, default=[1/64, 1/128])
    parser.add_argument("--cap", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("revision_results"))
    args = parser.parse_args()
    if not all(0 < q < 1 for q in args.q) or not all(np.isfinite(e) and e > 0 for e in args.epsilon) or args.cap < 0:
        parser.error("Require 0 < q < 1, positive finite epsilon and nonnegative cap.")
    rows = []
    for q in args.q:
        print(f"Computing both schedule optima for q={q}", flush=True)
        rows.extend(compare(q, args.epsilon, args.cap))
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out, "regular_matched", rows)
    print(args.out / "regular_matched.csv")


if __name__ == "__main__":
    main()
