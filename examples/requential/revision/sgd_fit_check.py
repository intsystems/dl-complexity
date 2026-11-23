"""Reanalyse existing SGD trajectories on the original update windows."""
import argparse
import csv
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from common import save


def basis(times, exponent):
    x = np.log(times / 512.)
    return x if exponent == 0 else np.expm1(exponent * x) / exponent


def errors(times, values, exponent):
    train = (times >= 512) & (times <= 2048)
    test = (times > 2048) & (times <= 4096)
    if train.sum() < 3 or test.sum() < 2:
        raise ValueError("Insufficient checkpoints for the specified windows")
    x = np.column_stack([np.ones(len(times)), basis(times, exponent)])
    coef = np.linalg.lstsq(x[train], values[train], rcond=None)[0]
    residual = x @ coef - values
    # Select beta using only the training curve; fit a slope/intercept to each run.
    return np.mean(residual[train]**2, axis=0), residual[test]


def analyse(rows):
    key = lambda r: (r["model"], int(r["batch"]), r["policy"])
    for (model, batch, policy), group in groupby(sorted(rows, key=key), key):
        series = {}
        for r in group:
            rep, time = int(r["replicate"]), int(r["T"])
            trajectory = series.setdefault(rep, {})
            if time in trajectory:
                raise ValueError("Duplicate observation")
            trajectory[time] = float(r["cum_kl_nats"])
        times = np.array(sorted(next(iter(series.values()))))
        if any(set(v) != set(times) for v in series.values()):
            raise ValueError("Mismatched checkpoint grids")
        values = np.array([[v[t] for t in times] for v in series.values()]).T
        # Fit beta separately within each trajectory to preserve independent pairs.
        grid = np.linspace(0., 1., 101)
        profiles = [errors(times, values, b) for b in grid]
        idx = np.argmin([x[0] for x in profiles], axis=0)
        adaptive = np.column_stack([profiles[i][1][:, j] for j, i in enumerate(idx)])
        residuals = {"log": profiles[0][1], "linear": profiles[-1][1], "power": adaptive}
        scores = {k: np.sqrt(np.mean(v**2, axis=0)) for k, v in residuals.items()}
        row = dict(model=model, batch=batch, policy=policy, replicates=len(idx),
                   beta_median=np.median(grid[idx]))
        for name, res in residuals.items():
            row[name + "_rmse_mean_curve"] = np.sqrt(np.mean(res.mean(axis=1)**2))
            row[name + "_mean_trajectory_rmse"] = scores[name].mean()
        for a, b in (("linear", "log"), ("power", "log"), ("power", "linear")):
            diff = scores[a] - scores[b]
            margin = (student_t.ppf(.975, len(diff)-1) * diff.std(ddof=1) / np.sqrt(len(diff))
                      if len(diff) > 1 else np.nan)
            name = a + "_minus_" + b
            row[name] = diff.mean()
            row[name + "_lo95"] = diff.mean() - margin
            row[name + "_hi95"] = diff.mean() + margin
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sgd_csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("revision_results"))
    args = parser.parse_args()
    with args.sgd_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(analyse(csv.DictReader(f)))
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out, "sgd_fit_check", rows)
    print(args.out / "sgd_fit_check.csv")


if __name__ == "__main__":
    main()
