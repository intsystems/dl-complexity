"""Independent controls for the PFR output law, optimization and SGD trajectory."""

import itertools
import math
import csv
from pathlib import Path
import numpy as np
from scipy.stats import binom
from common import LN2, bern_kl, bound, delta, read_delta, pfr, save
from forget import path_finder, decode


def validate(out):
    checks = []
    for i in list(range(1, 2049)) + [2**40 - 1]:
        assert read_delta(delta(i)) == (i, len(delta(i)))
    # Independent trials with nontrivial likelihood ratios and joint marks.
    samples = np.array([pfr(s, 0, 0.4, 0.65, 3)[1] for s in range(6000)])
    words = samples @ np.array([1, 2, 4])
    for j in range(8):
        prob = 0.65 ** j.bit_count() * 0.35 ** (3 - j.bit_count())
        error = (words == j).mean() - prob
        z = error / math.sqrt(prob * (1 - prob) / len(words))
        assert abs(z) < 5
        checks.append(dict(check="joint_pfr", case=j, standardized_error=z))
    # Exhaust all compositions of n=8; check both DP optima.
    with (out / "bayes.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["n"]) != 8:
                continue
            q, a = (0.37, 1.0) if row["model"] == "regular" else (0.0, 0.25)
            costs = []
            for mask in itertools.product([0, 1], repeat=7):
                n, cost = 0, 0.0
                for end in [i + 1 for i, x in enumerate(mask) if x] + [8]:
                    k = np.arange(n + 1)
                    d = bern_kl(q, (k + a) / (n + a + 1)) / LN2
                    cost += binom.pmf(k, n, q) @ bound((end - n) * d)
                    n = end
                costs.append(cost)
            np.testing.assert_allclose(
                min(costs), float(row["optimal_rhs_bits"]), atol=1e-10
            )
    # Enumerate all shorter bitstrings independently of the Dijkstra relaxations.
    for target in [12, 16, 20]:
        word, _ = path_finder(2026, "grid")(target, 0.0)
        for n in range(1, len(word)):
            for bits in itertools.product("01", repeat=n):
                try:
                    state = decode("".join(bits), 2026, "grid")
                except (IndexError, ValueError, AssertionError):
                    continue
                assert state != target
    # For x in {-1,1}, the exact error second moment obeys this scalar recursion.
    groups = {}
    with (out / "sgd.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] == "linear":
                groups.setdefault(
                    (int(r["batch"]), r["policy"], int(r["T"])), []
                ).append(float(r["cum_kl_nats"]))
    for (b, policy, t), values in groups.items():
        if t != max(key[2] for key in groups):
            continue
        moment, total = 0.8**2, 0.0
        for i in range(1, t + 1):
            total += moment / 2
            eta = 0.035 if policy == "constant" else 1 / (i + 9)
            moment = (1 - eta) ** 2 * moment + eta**2 / b
        z = (np.mean(values) - total) / (
            np.std(values, ddof=1) / math.sqrt(len(values))
        )
        assert abs(z) < 5
        checks.append(
            dict(check="linear_sgd", case=f"{b}_{policy}", standardized_error=z)
        )
    save(out, "validation", checks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    validate(parser.parse_args().out)
