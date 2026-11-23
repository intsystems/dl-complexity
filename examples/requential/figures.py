"""Render population-level summaries from the saved tables."""

import csv
from collections import defaultdict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(out):
    def read(name):
        with (out / (name + ".csv")).open(encoding="utf8") as f:
            return list(csv.DictReader(f))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    rows = read("schedules")
    for ax, model in zip(axes[0], ["regular", "quartic"]):
        for name in ["DP", "geom1.5", "geom2.0", "geom4.0", "geom8.0"]:
            group = [r for r in rows if r["model"] == model and r["policy"] == name]
            ax.plot(
                [-np.log2(float(r["epsilon_nats"])) for r in group],
                [float(r["rhs_bits"]) for r in group],
                ".-",
                label=name,
            )
        ax.set(
            title="Bayesian student · " + model,
            xlabel="−log₂ ε (ε in nats)",
            ylabel="Expected Eq. (2) bound, bits",
        )
        ax.legend(fontsize=8)
    for name, ax in [("sgd", axes[1, 0]), ("forgetting", axes[1, 1])]:
        grouped = defaultdict(lambda: defaultdict(list))
        for r in read(name):
            if name == "sgd":
                if r["model"] != "linear" or r["batch"] != "1":
                    continue
                grouped[r["policy"]][int(r["T"])].append(float(r["cum_kl_nats"]))
            elif r["model"] == "affine" and float(r["tolerance_nats"]) == 0.001:
                for key in ("trajectory_bits", "shortest_bits"):
                    grouped[key][int(r["T"])].append(int(r[key]))
        for key, series in grouped.items():
            x = sorted(series)
            y = np.array([np.mean(series[t]) for t in x])
            se = np.array(
                [np.std(series[t], ddof=1) / np.sqrt(len(series[t])) for t in x]
            )
            ax.plot(x, y, label=key.replace("_", " "))
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.16)
        ax.set_xscale("log", base=2)
        ax.set(
            title=(
                "Linear SGD · batch 1"
                if name == "sgd"
                else "Affine endpoint · δ′ = 0.001"
            ),
            xlabel="Updates T",
            ylabel="Cumulative KL, nats" if name == "sgd" else "Length, bits",
        )
        ax.legend()
    fig.savefig(out / "experiments.png", dpi=165)
    plt.close(fig)
