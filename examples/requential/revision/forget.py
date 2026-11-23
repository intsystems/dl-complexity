"""Shortest decodable path in a specified scalar linear-probability student."""

from functools import lru_cache
from heapq import heappush, heappop
import numpy as np
from common import bern_kl, proposals, pfr, replay, delta, read_delta, save

GRID = 32
START = GRID // 2


def update(state, y, model):
    if model == "affine":
        return (state + 4 + 24 * int(y)) / 2
    # Project one squared-loss SGD step with eta=1/4 onto a public finite grid.
    return min(GRID - 1, max(1, round((3 * state + GRID * int(y)) / 4)))


def decode(stream, seed, model):
    steps, pos = read_delta(stream)
    state = START
    for t in range(steps - 1):
        index, pos = read_delta(stream, pos)
        state = update(state, replay(seed, t, state / GRID, 1, index)[0], model)
    assert pos == len(stream)
    return state


def path_finder(seed, model):
    @lru_cache(None)
    def edges(t, state):
        rng = proposals(seed, t)
        first = {}
        i = 0
        while len(first) < 2:
            i += 1
            first.setdefault(bool(rng.random() < state / GRID), i)
        return [(update(state, y, model), delta(j)) for y, j in first.items()]

    @lru_cache(None)
    def shortest(target, tolerance):
        # Dijkstra cost includes the Elias horizon header. Its increments are nonnegative.
        heap = [(1, 0, START, "")]
        dist = {(0, START): 1}
        while heap:
            cost, t, state, body = heappop(heap)
            if cost != dist[t, state]:
                continue
            if bern_kl(target / GRID, state / GRID) <= tolerance:
                answer = delta(t + 1) + body
                assert len(answer) == cost and decode(answer, seed, model) == state
                return answer, state
            extra = len(delta(t + 2)) - len(delta(t + 1))
            for nxt, word in edges(t, state):
                length = cost + extra + len(word)
                if length < dist.get((t + 1, nxt), np.inf):
                    dist[t + 1, nxt] = length
                    heappush(heap, (length, t + 1, nxt, body + word))
        raise RuntimeError("target unreachable")

    return shortest


def experiment(out, steps, repeats, seed):
    rows = []
    times = set(np.geomspace(1, steps, 15).round().astype(int))
    for model in ("grid", "affine"):
        for rep in range(repeats):
            s = seed + rep
            shortest = path_finder(s, model)
            state, body = START, ""
            for t in range(steps):
                j, sample = pfr(s, t, state / GRID, 0.65)
                body += delta(j)
                state = update(state, sample[0], model)
                if t + 1 not in times:
                    continue
                trajectory = delta(t + 2) + body
                assert decode(trajectory, s, model) == state
                for tol in ((0.0, 0.001, 0.01) if model == "grid" else (0.001, 0.01)):
                    message, alternative = shortest(state, tol)
                    assert len(message) <= len(trajectory)
                    rows.append(
                        dict(
                            model=model,
                            seed=s,
                            T=t + 1,
                            tolerance_nats=tol,
                            target=state / GRID,
                            recovered=alternative / GRID,
                            distortion_nats=bern_kl(state / GRID, alternative / GRID),
                            trajectory_bits=len(trajectory),
                            shortest_bits=len(message),
                            gap_bits=len(trajectory) - len(message),
                            shortest_stream=message,
                        )
                    )
    save(out, "forgetting", rows)
