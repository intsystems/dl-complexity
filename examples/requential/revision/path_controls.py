"""Paired bit lengths for the original path, re-encoding, suffix and endpoint."""
import argparse
import csv
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count, groupby
from math import ceil, log
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from common import bern_kl, delta, pfr, proposals, read_delta, replay, save
from forget import START, GRID, decode, path_finder, update


def probability(t, state, model):
    return (1 + state) / (2 + t) if model == "mean" else state / GRID


def advance(state, label, model):
    return state + int(label) if model == "mean" else update(state, label, model)


def initial(model):
    return 0 if model == "mean" else START


def first_word(seed, t, p, label):
    rng = proposals(seed, t)
    for j in count(1):
        if (rng.random() < p) == label:
            return delta(j)


def encode_labels(labels, seed, model):
    state = initial(model)
    words = [delta(len(labels) + 1)]
    for t, y in enumerate(labels):
        words.append(first_word(seed, t, probability(t, state, model), y))
        state = advance(state, y, model)
    return "".join(words)


def decoded_probability(stream, seed, model):
    if model != "mean":
        return decode(stream, seed, model) / GRID
    header, pos = read_delta(stream)
    state = 0
    for t in range(header - 1):
        j, pos = read_delta(stream, pos)
        y = replay(seed, t, probability(t, state, model), 1, j)[0]
        state += int(y)
    if pos != len(stream):
        raise ValueError("Trailing code words")
    return probability(header - 1, state, model)


def mean_solver(seed):
    @lru_cache(None)
    def edges(t, ones):
        p = probability(t, ones, "mean")
        return [(ones + y, first_word(seed, t, p, y)) for y in (0, 1)]

    def shortest(target, tol):
        heap = [(1, 0, 0, "")]
        dist = {(0, 0): 1}
        while heap:
            bits, t, ones, body = heappop(heap)
            if bits != dist[t, ones]:
                continue
            if bern_kl(target, probability(t, ones, "mean")) <= tol:
                return delta(t + 1) + body
            header_change = len(delta(t + 2)) - len(delta(t + 1))
            for nxt, word in edges(t, ones):
                length = bits + header_change + len(word)
                if length < dist.get((t + 1, nxt), np.inf):
                    dist[t + 1, nxt] = length
                    heappush(heap, (length, t + 1, nxt, body + word))
        raise ValueError("Unreachable target")
    return shortest


def control_rows(seed, model, times, tol):
    find = mean_solver(seed) if model == "mean" else path_finder(seed, model)
    h = max(0, ceil((log(9 / 7) - log(tol)) / log(4)))
    labels, original, first = [], "", ""
    state = initial(model)
    for t in range(max(times)):
        p = probability(t, state, model)
        j, sample = pfr(seed, t, p, .65)
        y = bool(sample[0])
        original += delta(j)
        first += first_word(seed, t, p, y)
        labels.append(y)
        state = advance(state, y, model)
        end = t + 1
        if end not in times:
            continue
        target = probability(end, state, model)
        shortest = find(target, tol) if model == "mean" else find(state, tol)[0]
        suffix = encode_labels(labels[max(0, end - h):], seed, model)
        header = delta(end + 1)
        words = {"trajectory": header + original, "same_labels": header + first,
                 "suffix": suffix, "shortest": shortest}
        row = dict(model=model, seed=seed, T=end, tolerance_nats=tol, target=target,
                   suffix_steps=min(end, h))
        for name, stream in words.items():
            recovered = decoded_probability(stream, seed, model)
            loss = float(bern_kl(target, recovered))
            if name in ("trajectory", "same_labels") and recovered != target:
                raise AssertionError("Full label path changed the endpoint")
            row[name + "_bits"] = len(stream)
            row[name + "_kl_nats"] = loss
        valid = row["suffix_kl_nats"] <= tol
        if model == "affine" and not valid:
            raise AssertionError("Affine suffix exceeded its proven tolerance")
        if (row["shortest_kl_nats"] > tol or len(shortest) > len(header + first)
                or len(first) > len(original) or (valid and len(shortest) > len(suffix))):
            raise AssertionError("Endpoint comparison failed")
        row.update(suffix_feasible=int(valid), recoding_gain_bits=len(original) - len(first),
                   endpoint_gain_after_recoding_bits=len(header + first) - len(shortest))
        yield row


def summaries(rows):
    key = lambda r: (r["model"], r["T"], r["tolerance_nats"])
    for (model, end, tol), group in groupby(sorted(rows, key=key), key):
        group = list(group)
        summary = dict(model=model, T=end, tolerance_nats=tol, repeats=len(group),
                       suffix_feasible_fraction=np.mean([r["suffix_feasible"] for r in group]))
        for name in ("trajectory_bits", "same_labels_bits", "suffix_bits", "shortest_bits",
                     "suffix_kl_nats", "recoding_gain_bits", "endpoint_gain_after_recoding_bits"):
            values = np.array([r[name] for r in group])
            summary[name + "_mean"] = values.mean()
            if name in ("recoding_gain_bits", "endpoint_gain_after_recoding_bits"):
                half = (student_t.ppf(.975, len(group) - 1) * values.std(ddof=1) / np.sqrt(len(group))
                        if len(group) > 1 else np.nan)
                summary[name + "_lo95"] = values.mean() - half
                summary[name + "_hi95"] = values.mean() + half
        yield summary


def check_old(rows, filename):
    with filename.open(encoding="utf-8-sig", newline="") as f:
        old = {(r["model"], int(r["seed"]), int(r["T"]), float(r["tolerance_nats"])): r
               for r in csv.DictReader(f)}
    for r in rows:
        if r["model"] == "mean":
            continue
        key = (r["model"], r["seed"], r["T"], r["tolerance_nats"])
        a = old[key]
        if (float(a["target"]) != r["target"] or
                any(int(a[k]) != r[k] for k in ("shortest_bits", "trajectory_bits"))):
            raise ValueError(f"Original results differ for {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("affine", "grid", "mean"), default=["affine", "grid", "mean"])
    parser.add_argument("--times", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=48)
    parser.add_argument("--tol", type=float, default=.001)
    parser.add_argument("--check-against", type=Path)
    parser.add_argument("--out", type=Path, default=Path("revision_results"))
    args = parser.parse_args()
    if args.check_against is not None and not args.check_against.is_file():
        parser.error(f"Reference CSV not found: {args.check_against.resolve()}. "
                     "From the requential directory use --check-against results/forgetting.csv")
    if not np.isfinite(args.tol) or args.tol <= 0 or args.repeats < 1 or min(args.times) < 1 or args.seed < 0:
        parser.error("Require positive tolerance, repeats and times and a nonnegative seed.")
    rows = []
    for model in dict.fromkeys(args.models):
        print(f"Computing paired paths: {model}", flush=True)
        for seed in range(args.seed, args.seed + args.repeats):
            rows.extend(control_rows(seed, model, set(args.times), args.tol))
    if args.check_against:
        check_old(rows, args.check_against)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out, "path_controls", rows)
    save(args.out, "path_summary", list(summaries(rows)))
    print(args.out / "path_summary.csv")


if __name__ == "__main__":
    main()
