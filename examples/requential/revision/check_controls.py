"""Small deterministic fixtures; no study runs."""
from itertools import product
from math import prod

import numpy as np

from common import bern_kl, delta
from path_controls import (control_rows, decoded_probability, encode_labels, mean_solver,
                           summaries)
from quartic_pfr_cost import message_bits
from regular_pfr import compare, log_index_mean, problem


def check_cost():
    for batch in (1, 3, 4):
        marks = list(product((0, 1), repeat=batch))
        for p, q in ((.12, .37), (.8, .37), (.5, .5)):
            proposal = np.array([prod(p if y else 1-p for y in mark) for mark in marks])
            teacher = np.array([prod(q if y else 1-q for y in mark) for mark in marks])
            ratio = teacher / proposal
            for mark, r in zip(marks, ratio):
                mean = 1 + sum(proposal * np.maximum(0., r - ratio))
                log_mean = log_index_mean(batch, sum(mark), np.array(p), q)
                np.testing.assert_allclose(np.exp(log_mean), mean, atol=1e-12)
            if p == .5:
                np.testing.assert_allclose(message_bits(0), 1)
    # Direct sum over indices, independent of the threshold-series implementation.
    for mean in (1.1, 4.4, 19.0):
        bits = sum(len(delta(j)) * (1 / mean) * (1 - 1/mean)**(j-1) for j in range(1, 1500))
        np.testing.assert_allclose(message_bits(np.log(mean)), bits, atol=1e-12)


def check_dp():
    from regular_pfr import optimizers, recover
    risk, costs = problem(.37)
    n = 4
    totals = []
    for interior in product((False, True), repeat=n-1):
        cuts = [0] + [i+1 for i, choose in enumerate(interior) if choose] + [n]
        totals.append(sum(costs(a, b-a) for a, b in zip(cuts, cuts[1:])))
    previous = optimizers(n, costs)
    for column in (0, 1):
        start, total = 0, 0.
        for b in recover(previous, n, column):
            total += costs(start, b)[column]
            start += b
        np.testing.assert_allclose(total, np.min(totals, axis=0)[column])
    assert all(r["n"] == 0 and r["expected_pfr_bits"] == 0 for r in compare(.37, [risk(0)], 0))


def check_paths():
    for model in ("affine", "grid", "mean"):
        rows = list(control_rows(5, model, {1, 3}, .001))
        assert len(rows) == 2 and len(list(summaries(rows))) == 2
    seed, target, tol = 5, 2/3, .001
    best = mean_solver(seed)(target, tol)
    options = []
    for steps in range(len(best)):
        if steps + len(delta(steps + 1)) > len(best):
            break
        for labels in product((0, 1), repeat=steps):
            msg = encode_labels(labels, seed, "mean")
            if bern_kl(target, decoded_probability(msg, seed, "mean")) <= tol:
                options.append(len(msg))
    assert len(best) == min(options)
    # A last-six suffix always satisfies the bound in the affine model.
    for labels in product((0, 1), repeat=7):
        full = encode_labels(labels, seed, "affine")
        suffix = encode_labels(labels[-6:], seed, "affine")
        kl = bern_kl(decoded_probability(full, seed, "affine"), decoded_probability(suffix, seed, "affine"))
        assert kl <= 9/7 * 4**-6


if __name__ == "__main__":
    check_cost()
    check_dp()
    check_paths()
    print("OK: conditional law, geometric code, schedule optimization, decoding, shortest paths and suffix bound.")
