"""Compare causal blockwise, online and replay codes on a small CPU MLP.

After installation: python examples/prequential_demo.py --json result.json
All six runs use identical ordered data and initial weights. Calibration may
improve or worsen the code; this example does not assert a universal ranking.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from dl_complexity import Provenance
from dl_complexity.coding import (
    BlockwisePrequential,
    OnlinePrequential,
    ReplayPrequential,
    TemperatureScaler,
    prepare_blockwise,
    prepare_online,
    prepare_replay,
)


@dataclass(frozen=True)
class DemoConfig:
    """Training and coding settings recorded alongside the measured lengths."""

    n: int = 192
    seed: int = 7
    initial_count: int = 16
    batch_size: int = 16
    train_steps: int = 80
    update_steps: int = 10
    replay_size: int = 32
    calibration_steps: int = 15
    learning_rate: float = 0.03

    def __post_init__(self):
        for name in (
            "n", "initial_count", "batch_size", "train_steps",
            "update_steps", "calibration_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.initial_count >= self.n:
            raise ValueError("initial_count must be smaller than n")
        for name in ("seed", "replay_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass
class Learner:
    """Keep optimizer history in the state shared between stream updates."""

    model: nn.Module
    optimizer: torch.optim.Optimizer


def make_data(config):
    """Generate a fixed binary classification sequence using a local RNG."""
    generator = torch.Generator().manual_seed(config.seed)
    x = torch.randn(config.n, 2, generator=generator, dtype=torch.float64)
    logits = 2.5 * x[:, 0] - 1.5 * x[:, 1]
    y = (torch.rand(config.n, generator=generator) < logits.sigmoid()).long()
    return x, y


def make_learner(config):
    """Start each run/prefix with identical weights, preserving the caller RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed + 1)
        model = nn.Sequential(
            nn.Linear(2, 8, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(8, 2, dtype=torch.float64),
        )
    return Learner(
        model,
        torch.optim.Adam(model.parameters(), lr=config.learning_rate),
    )


def predict_logits(state, batch):
    """Explicit eval adapter for a custom state containing a model and optimizer."""
    state.model.eval()
    with torch.no_grad():
        return state.model(batch)


def fit(state, batch, labels, steps):
    """Update only on the revealed examples supplied by a preparation callback."""
    state.model.train()
    with torch.enable_grad():
        for _ in range(steps):
            state.optimizer.zero_grad()
            F.cross_entropy(state.model(batch), labels).backward()
            state.optimizer.step()


def block_boundaries(n, initial_count):
    """Double prefix endpoints, including a final possibly shorter block."""
    boundaries = [initial_count]
    while boundaries[-1] < n:
        boundaries.append(min(2 * boundaries[-1], n))
    return boundaries


def run_demo(config=None):
    """Run six protocols and return JSON-compatible settings and causal ledgers.

    The elapsed time includes preparation/training and ledger evaluation, but
    excludes generating the shared data. No test set or external files are read.
    """
    config = DemoConfig() if config is None else config
    x, y = make_data(config)

    def factory():
        return make_learner(config)

    def train(state, prefix_x, prefix_y):
        fit(state, prefix_x, prefix_y, config.train_steps)

    def update(state, new_x, new_y):
        fit(state, new_x, new_y, config.update_steps)

    def replay_update(state, new_x, new_y, replay_x, replay_y):
        # Replay affects the learner, never the number of labels charged.
        fit(
            state,
            torch.cat((new_x, replay_x)),
            torch.cat((new_y, replay_y)),
            config.update_steps,
        )

    def calibrator_factory():
        return TemperatureScaler(steps=config.calibration_steps)

    rows = []
    for protocol, estimator in (
        ("blockwise", BlockwisePrequential),
        ("online", OnlinePrequential),
        ("replay", ReplayPrequential),
    ):
        for calibrated in (False, True):
            started = perf_counter()
            shared = dict(
                model_factory=factory,
                predict_logits=predict_logits,
                n_classes=2,
                provenance=Provenance(
                    state_id=f"mlp-2-8-2-seed-{config.seed + 1}",
                    data_id=f"prequential-demo-n{config.n}-seed-{config.seed}",
                    likelihood_id="categorical",
                ),
            )
            if protocol == "blockwise":
                ledger = prepare_blockwise(
                    x, y,
                    boundaries=block_boundaries(config.n, config.initial_count),
                    train=train,
                    calibrator_factory=calibrator_factory if calibrated else None,
                    **shared,
                )
            else:
                stream = dict(
                    batch_size=config.batch_size,
                    initial_count=config.initial_count,
                    calibrator=calibrator_factory() if calibrated else None,
                    **shared,
                )
                if protocol == "online":
                    ledger = prepare_online(x, y, update=update, **stream)
                else:
                    ledger = prepare_replay(
                        x, y, update=replay_update,
                        replay_size=config.replay_size, seed=config.seed + 2,
                        **stream,
                    )
            result = estimator(unit="bits").estimate(inputs=ledger)
            rows.append({
                "protocol": protocol,
                "calibrated": calibrated,
                "total_bits": result.primary.value.item(),
                "initial_bits": result.quantities["initial"].value.item(),
                "predictive_bits": result.quantities["predictive"].value.item(),
                "bits_per_observation": (
                    result.quantities["per_observation"].value.item()
                ),
                "elapsed_seconds": perf_counter() - started,
                "provenance": asdict(result.provenance),
                "ledger": {
                    "n": ledger.n,
                    "initial_count": ledger.initial_count,
                    "initial_nll": float(ledger.initial_nll),
                    "entries": [
                        {**asdict(entry), "nll": float(entry.nll)}
                        for entry in ledger.entries
                    ],
                },
            })
    return {
        "config": asdict(config),
        "torch_version": torch.__version__,
        "device": "cpu",
        "dtype": "float64",
        "model": "MLP 2-8-2, tanh",
        "initialization_seed": config.seed + 1,
        "replay_seed": config.seed + 2,
        "uniform_bits": float(config.n),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write settings, scores and ledgers")
    for name in ("n", "seed", "initial_count", "batch_size", "train_steps",
                 "update_steps", "replay_size", "calibration_steps"):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int,
            default=getattr(DemoConfig(), name),
        )
    parser.add_argument(
        "--learning-rate", type=float, default=DemoConfig().learning_rate
    )
    args = vars(parser.parse_args())
    output = args.pop("json")
    try:
        config = DemoConfig(**args)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    report = run_demo(config)
    print(f"CPU MLP 2-8-2, n={config.n}, seed={config.seed}")
    print(f"Uniform label code: {report['uniform_bits']:.1f} bits")
    print("protocol   calibrated  total_bits  initial_bits  predictive_bits  bits/obs")
    for row in report["rows"]:
        print(
            f"{row['protocol']:<10} {str(row['calibrated']):<10} "
            f"{row['total_bits']:>10.4f} {row['initial_bits']:>13.4f} "
            f"{row['predictive_bits']:>16.4f} {row['bits_per_observation']:>9.4f}"
        )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
