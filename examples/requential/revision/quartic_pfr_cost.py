"""Expected Elias-delta length of exact PFR for q=0, p=w**4 and uniform w."""

import argparse
import math

import numpy as np


def message_bits(log_mean):
    """Geometric index with success probability exp(-log_mean); no sampling."""
    h = np.atleast_1d(np.asarray(log_mean, dtype=float))
    if np.any(~np.isfinite(h)) or np.any(h < 0):
        raise ValueError("Expected a finite nonnegative negative log probability.")
    # log(-log(1-exp(-h))); its large-h limit -h avoids underflow.
    log_decay = -h.copy()
    mask = (h > 0) & (h <= 1)
    log_decay[mask] = np.log(-np.log(-np.expm1(-h[mask])))
    # Near h=35 forming 1-exp(-h) loses precision; log1p is stable there.
    mask = (h > 1) & (h < 35)
    log_decay[mask] = np.log(-np.log1p(-np.exp(-h[mask])))
    log_decay[h == 0] = np.inf
    result = np.ones_like(h)
    k = 1
    while True:
        # P(J >= 2**k) = (1-exp(-h))**(2**k - 1).
        log_threshold = k * math.log(2) + math.log1p(-math.exp(-k * math.log(2)))
        exponent = log_threshold + log_decay
        survival = np.exp(-np.exp(np.minimum(exponent, 7.0)))
        jump = 3 if ((k + 1) & k) == 0 else 1
        result += jump * survival
        # Subsequent tails are bounded by successive squares of this probability.
        if survival.max() < 1e-16:
            return result
        k += 1


def edge(n, batch):
    return message_bits(batch * np.log1p(.25 / (n + 1)))


def optimum(target):
    cost = np.full(target + 1, np.inf)
    cost[0] = 0
    previous = np.zeros(target + 1, dtype=int)
    for n in range(target):
        ends = np.arange(n + 1, target + 1)
        candidate = cost[n] + edge(n, ends - n)
        improved = candidate < cost[ends]
        cost[ends[improved]] = candidate[improved]
        previous[ends[improved]] = n
    sizes = []
    n = target
    while n:
        sizes.append(int(n - previous[n]))
        n = int(previous[n])
    return cost[target], sizes[::-1]


def schedule_bits(sizes):
    n, total = 0, 0.
    for size in sizes:
        total += edge(n, size)[0]
        n += size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, default=1 / 256)
    args = parser.parse_args()
    if not 0 < args.epsilon <= 1:
        parser.error("Choose 0 < epsilon <= 1 (nats).")
    target = max(0, math.ceil(.25 / math.expm1(args.epsilon) - 1))
    bits, sizes = optimum(target)
    print(f"n={target}, batches={sizes}, expected bits={bits:.8f}")
    print(f"Final KL (nats): {math.log1p(.25 / (target+1)):.8g}")
    if target == 63:
        print(f"Expected bits for [7, 56]: {schedule_bits([7,56]):.8f}")


if __name__ == "__main__":
    main()
