"""All KLs returned by bern_kl are in nats. Message lengths are in bits."""

import csv
from pathlib import Path
import numpy as np
from scipy.special import rel_entr

LN2 = np.log(2.0)
BETA = 1 + 1 / (np.e * LN2)
KAPPA = 1 + BETA + 2 * np.log2(1 + BETA)


def bern_kl(q, p):
    return rel_entr(q, p) + rel_entr(1 - q, 1 - p)


def bound(bits):
    return bits + 2 * np.log1p(bits) / LN2 + KAPPA


def delta(i):
    """Elias delta for a strictly positive integer."""
    word = format(i, "b")
    size = format(len(word), "b")
    return "0" * (len(size) - 1) + size + word[1:]


def read_delta(stream, offset=0):
    start = stream.index("1", offset)
    stop = start + start - offset + 1
    width = int(stream[start:stop], 2)
    end = stop + width - 1
    if end > len(stream):
        raise ValueError("truncated Elias word")
    return int("1" + stream[stop:end], 2), end


def proposals(seed, t):
    return np.random.default_rng([seed, t, 1])


def pfr(seed, t, p, q, batch=1):
    """Poisson race on iid Bernoulli batches. No candidate cap."""
    marks = proposals(seed, t)
    clock = np.random.default_rng([seed, t, 2])
    # Use log scores so that unlikely individual batches remain representable.
    max_log_ratio = batch * np.log(max(q / p, (1 - q) / (1 - p)))
    time, best, i = 0.0, np.inf, 0
    while True:
        time += clock.exponential()
        log_time = np.log(time)
        if log_time - max_log_ratio >= best:
            return index, selected
        i += 1
        sample = marks.random(batch) < p
        count = int(sample.sum())
        if (count and q == 0) or (count < batch and q == 1):
            continue
        ratio = count * np.log(q / p) if count else 0.0
        if count < batch:
            ratio += (batch - count) * np.log((1 - q) / (1 - p))
        if log_time - ratio < best:
            best, index, selected = log_time - ratio, i, sample


def replay(seed, t, p, batch, index):
    marks = proposals(seed, t)
    for _ in range(index):
        sample = marks.random(batch) < p
    return sample


def save(out, name, rows):
    with (Path(out) / (name + ".csv")).open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
