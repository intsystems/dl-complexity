"""Measure paired S(2T)-S(T) for rank-zero factorized regression, B=1."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def simulate(steps, repeats, seed):
    # Keep initialization, random draw order and pre-update loss identical to sgd.py.
    rng = np.random.default_rng(seed)
    support = np.sqrt(2) * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    u = np.array([[0.3], [0.0]]) + rng.normal(0, 0.15, (repeats, 2, 1))
    v = np.array([[0.3, 0.0]]) + rng.normal(0, 0.15, (repeats, 1, 2))
    cumulative = np.zeros(repeats)
    snapshots = {}
    for step in range(1, steps + 1):
        residual = support @ (u @ v)  # True mean is zero at every input.
        cumulative += 0.5 * np.sum(residual**2, axis=2).mean(axis=1)
        if (step & (step - 1)) == 0:
            snapshots[step] = cumulative.copy()
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
    return snapshots


def compare_previous(folder, snapshots, seed, repeats):
    config = json.loads((folder / "run.json").read_text(encoding="utf8"))
    if config["seed"] != seed or config["repeats"] != repeats:
        raise ValueError("Prefix check requires matching --seed and --repeats.")
    horizon = config["steps"]
    if horizon not in snapshots:
        raise ValueError("Previous horizon must be a power of two within the new run.")
    with (folder / "sgd.csv").open(encoding="utf8", newline="") as f:
        rows = sorted(
            (int(r["replicate"]), float(r["cum_kl_nats"]))
            for r in csv.DictReader(f)
            if r["model"] == "rank0"
            and r["policy"] == "inverse"
            and int(r["batch"]) == 1
            and int(r["T"]) == horizon
        )
    if [rep for rep, _ in rows] != list(range(repeats)):
        raise ValueError("Missing or duplicate trajectories in the previous run.")
    np.testing.assert_allclose(
        snapshots[horizon], [v for _, v in rows], rtol=1e-11, atol=1e-12
    )
    return horizon


def summarize(snapshots):
    summary, trajectories = [], []
    prev = None
    for start in sorted(snapshots):
        if 2 * start not in snapshots:
            continue
        end = 2 * start
        delta = snapshots[end] - snapshots[start]
        n = len(delta)
        critical = student_t.ppf(0.975, n - 1)
        mean = delta.mean()
        se = delta.std(ddof=1) / np.sqrt(n)
        row = dict(
            T=start,
            two_T=end,
            mean_delta_nats=mean,
            ci95_low=mean - critical * se,
            ci95_high=mean + critical * se,
            local_slope=mean / np.log(2),
            reference_delta_nats=np.log(2),
            paired_change_nats=None,
            change_ci95_low=None,
            change_ci95_high=None,
        )
        if prev is not None:
            change = delta - prev
            margin = critical * change.std(ddof=1) / np.sqrt(n)
            row.update(
                paired_change_nats=change.mean(),
                change_ci95_low=change.mean() - margin,
                change_ci95_high=change.mean() + margin,
            )
        summary.append(row)
        for rep, increment in enumerate(delta):
            trajectories.append(
                dict(
                    replicate=rep,
                    T=start,
                    two_T=end,
                    S_T=snapshots[start][rep],
                    S_two_T=snapshots[end][rep],
                    delta_nats=increment,
                )
            )
        prev = delta
    return summary, trajectories


def save_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(path, rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    end = [r["two_T"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(
        end, [r["mean_delta_nats"] for r in rows], "o-", label="Mean paired increment"
    )
    ax.fill_between(
        end,
        [r["ci95_low"] for r in rows],
        [r["ci95_high"] for r in rows],
        alpha=0.2,
        label="Pointwise 95% CI",
    )
    ax.axhline(np.log(2), color="black", ls="--", label="Reference: ln 2")
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="End of doubling interval (2T)",
        ylabel="S(2T) − S(T), nats",
        title="Rank-zero regression · inverse step size · batch 1",
    )
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("rank0_results"))
    parser.add_argument(
        "--check-against",
        type=Path,
        help="Previous results folder for a paired prefix check.",
    )
    args = parser.parse_args()
    if (
        args.steps < 2
        or args.steps & (args.steps - 1)
        or args.repeats < 2
        or args.seed < 0
    ):
        parser.error("Use a power-of-two horizon >= 2, repeats >= 2 and seed >= 0.")
    snapshots = simulate(args.steps, args.repeats, args.seed)
    verified = None
    if args.check_against:
        verified = compare_previous(
            args.check_against, snapshots, args.seed, args.repeats
        )
    rows, trajectories = summarize(snapshots)
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "doubling_summary.csv", rows)
    save_csv(args.out / "trajectories.csv", trajectories)
    plot(args.out / "doubling.png", rows)
    config = dict(vars(args), verified_prefix_steps=verified, numpy=np.__version__)
    (args.out / "run.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf8"
    )
    print(f"Saved {len(rows)} doubling intervals to {args.out.resolve()}")


if __name__ == "__main__":
    main()
