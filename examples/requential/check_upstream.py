"""Check actual decoded Task 1 streams against the upstream OnlinePrequential API."""

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.special import betaln
import torch
from dl_complexity.coding import OnlinePrequential, prepare_online
from common import LN2, read_delta, replay, save


def compare(out):
    rows = []
    with (out / "pfr.csv").open(encoding="utf8") as file:
        for row in csv.DictReader(file):
            a = 1 if row["model"] == "regular" else 0.25
            sizes = [1] * 16 if row["policy"] == "unit" else [1, 2, 4, 8]
            seq = []
            ones = offset = 0
            for t, b in enumerate(sizes):
                j, offset = read_delta(row["message"], offset)
                y = replay(int(row["seed"]), t, (a + ones) / (a + len(seq) + 1), b, j)
                seq.extend(y.tolist())
                ones += int(y.sum())
            assert len(seq) == int(row["n"]) and ones == int(row["ones"])

            def predict(state, x):
                p = (state[0] + a) / (sum(state) + a + 1)
                return torch.tensor([[np.log1p(-p), np.log(p)]], dtype=torch.float64)

            def update(state, x, y):
                state[0] += int(y.sum())
                state[1] += len(y) - int(y.sum())

            y = torch.tensor(seq, dtype=torch.long)
            ledger = prepare_online(
                torch.arange(len(y))[:, None],
                y,
                model_factory=lambda: [0, 0],
                predict_logits=predict,
                update=update,
                n_classes=2,
            )
            bits = float(
                OnlinePrequential(unit="bits").estimate(inputs=ledger).primary.value
            )
            exact = (betaln(a, 1) - betaln(a + ones, 1 + len(seq) - ones)) / LN2
            np.testing.assert_allclose(bits, exact, atol=1e-6)
            rows.append(
                dict(
                    model=row["model"],
                    seed=row["seed"],
                    policy=row["policy"],
                    upstream_online_bits=bits,
                    bayes_bits=exact,
                    absolute_error=abs(exact - bits),
                )
            )
    save(out, "upstream_check", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    compare(parser.parse_args().out)
