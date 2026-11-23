"""Run all three experiments on CPU, with independent controls for each budget."""

import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import bayes
import sgd
import forget

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=48)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--dp-limit", type=int, default=512)
    parser.add_argument("--forget-steps", type=int, default=128)
    args = parser.parse_args()
    if (
        args.repeats < 2
        or args.steps < 64
        or min(args.dp_limit, args.forget_steps) < 1
        or args.seed < 0
    ):
        parser.error("Use repeats >= 2, steps >= 64 and positive horizons.")
    args.out.mkdir(parents=True, exist_ok=True)
    bayes.experiment(args.out, args.dp_limit, args.repeats, args.seed)
    sgd.experiment(args.out, args.steps, args.repeats, args.seed)
    forget.experiment(args.out, args.forget_steps, args.repeats, args.seed)
    from figures import plot

    plot(args.out)
    metadata = dict(
        vars(args),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
    )
    (args.out / "run.json").write_text(
        json.dumps(metadata, default=str, indent=2), encoding="utf8"
    )
