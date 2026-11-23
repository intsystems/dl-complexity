"""Paired rank-zero regression: compare noisy SGD with exact population descent."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from rank0_doubling import save_csv


def norm_sq(factor):
    return np.sum(factor**2, axis=(1, 2))


def population_step(u, v, eta):
    # K = ||u||² ||v||² / 2. Both gradients use the same pre-update factors.
    return (1 - eta * norm_sq(v)[:, None, None]) * u, (
        1 - eta * norm_sq(u)[:, None, None]
    ) * v


def observe(u, v):
    a, b = norm_sq(u), norm_sq(v)
    return np.sqrt(a), np.sqrt(b), 0.5 * a * b


def histories(steps, repeats, seed):
    rng = np.random.default_rng(seed)
    support = np.sqrt(2) * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    u = np.array([[0.3], [0.0]]) + rng.normal(0, 0.15, (repeats, 2, 1))
    v = np.array([[0.3, 0.0]]) + rng.normal(0, 0.15, (repeats, 1, 2))
    ug, vg = u.copy(), v.copy()
    totals = np.zeros((2, repeats))
    # Index h means h updates completed. S(h) sums h pre-update losses.
    snapshots = {0: (totals.copy(), observe(u, v), observe(ug, vg))}
    for step in range(1, steps + 1):
        residual = support @ (u @ v)
        totals[0] += 0.5 * np.sum(residual**2, axis=2).mean(axis=1)
        totals[1] += 0.5 * norm_sq(ug) * norm_sq(vg)
        ix = rng.integers(len(support), size=(repeats, 1))
        x = support[ix]
        error = residual[np.arange(repeats)[:, None], ix] - rng.normal(
            size=(repeats, 1, 2)
        )
        hidden = x @ u
        back = error @ v.swapaxes(1, 2)
        eta = 1 / (step + 9)
        u -= eta * (x.swapaxes(1, 2) @ back)
        v -= eta * (hidden.swapaxes(1, 2) @ error)
        ug, vg = population_step(ug, vg, eta)
        if (step & (step - 1)) == 0:
            snapshots[step] = (totals.copy(), observe(u, v), observe(ug, vg))
    return snapshots


def verify_sgd(folder, history, repeats, seed):
    config = json.loads((folder / "run.json").read_text(encoding="utf8"))
    if (config["seed"], config["repeats"]) != (seed, repeats):
        raise ValueError(
            "Previous seed and repeat count must match for paired verification."
        )
    horizon = min(max(history), config["steps"])
    expected = {
        (rep, end) for rep in range(repeats) for end in history if 2 <= end <= horizon
    }
    with (folder / "trajectories.csv").open(encoding="utf8", newline="") as f:
        for row in csv.DictReader(f):
            start, end, rep = int(row["T"]), int(row["two_T"]), int(row["replicate"])
            if end > horizon:
                continue
            if end != 2 * start or (rep, end) not in expected:
                raise ValueError("Unexpected or duplicate previous trajectory record.")
            expected.remove((rep, end))
            np.testing.assert_allclose(
                [history[start][0][0, rep], history[end][0][0, rep]],
                [float(row["S_T"]), float(row["S_two_T"])],
                rtol=1e-11,
                atol=1e-12,
            )
    if expected:
        raise ValueError("Previous trajectory table is incomplete.")
    return horizon


def summarize(history):
    summaries, checkpoints = [], []
    for end, (totals, stochastic, exact) in sorted(history.items()):
        for method, cumulative, state in zip(
            ("sgd", "population"), totals, (stochastic, exact)
        ):
            for rep, (u, v, kl) in enumerate(zip(*state)):
                checkpoints.append(
                    dict(
                        updates=end,
                        replicate=rep,
                        method=method,
                        u_norm=u,
                        v_norm=v,
                        squared_norm_difference=u * u - v * v,
                        endpoint_kl_nats=kl,
                        cumulative_kl_nats=cumulative[rep],
                    )
                )
        if end < 2:
            continue
        sgd, gd = totals - history[end // 2][0]
        gap = sgd - gd
        row = dict(T=end // 2, two_T=end, pairs=len(gap))
        for name, values in (
            ("sgd", sgd),
            ("population", gd),
            ("paired_sgd_minus_population", gap),
        ):
            mean = values.mean()
            margin = (
                student_t.ppf(0.975, len(values) - 1)
                * values.std(ddof=1)
                / np.sqrt(len(values))
            )
            row[name + "_mean_nats"] = mean
            row[name + "_ci95_low"] = mean - margin
            row[name + "_ci95_high"] = mean + margin
        summaries.append(row)
    return summaries, checkpoints


def plot(out, rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout="constrained")
    x = [r["two_T"] for r in rows]
    for name, label, ax in (
        ("sgd", "SGD", axes[0]),
        ("population", "Population GD", axes[0]),
        ("paired_sgd_minus_population", "SGD minus population GD", axes[1]),
    ):
        (line,) = ax.plot(x, [r[name + "_mean_nats"] for r in rows], ".-", label=label)
        ax.fill_between(
            x,
            [r[name + "_ci95_low"] for r in rows],
            [r[name + "_ci95_high"] for r in rows],
            color=line.get_color(),
            alpha=0.15,
        )
    axes[0].axhline(np.log(2), ls="--", color="gray", label="ln 2")
    axes[1].axhline(0, ls="--", color="gray")
    for ax in axes:
        ax.legend()
        ax.set_xscale("log", base=2)
    axes[0].set_ylabel("KL increment (nats)")
    axes[1].set(
        xlabel="End of doubling interval (2T)", ylabel="Paired difference (nats)"
    )
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--check-against", type=Path, help="Output folder from rank0_doubling.py"
    )
    parser.add_argument("--out", type=Path, default=Path("rank0_mechanism_results"))
    args = parser.parse_args()
    if (
        args.steps < 2
        or args.steps & (args.steps - 1)
        or args.repeats < 2
        or args.seed < 0
    ):
        parser.error("Use a power-of-two horizon >= 2, repeats >= 2 and seed >= 0.")
    history = histories(args.steps, args.repeats, args.seed)
    verified = (
        verify_sgd(args.check_against, history, args.repeats, args.seed)
        if args.check_against
        else None
    )
    rows, states = summarize(history)
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "paired_summary.csv", rows)
    save_csv(args.out / "factor_history.csv", states)
    plot(args.out, rows)
    config = dict(vars(args), verified_sgd_steps=verified, numpy=np.__version__)
    (args.out / "run.json").write_text(
        json.dumps(config, default=str, indent=2), encoding="utf8"
    )
    print(f"Saved results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
