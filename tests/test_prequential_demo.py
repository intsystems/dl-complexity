"""Exercise the documented MLP protocols, causal ledgers and reproducibility."""

import json
import math
import runpy
from pathlib import Path

import pytest
import torch

from dl_complexity.coding import (
    BlockwisePrequential,
    OnlinePrequential,
    PrequentialEntry,
    PrequentialLedger,
    ReplayPrequential,
)


@pytest.fixture(scope="module")
def demo():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "prequential_demo.py")
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield namespace
    finally:
        torch.set_num_threads(previous_threads)


@pytest.fixture(scope="module")
def report(demo):
    config = demo["DemoConfig"](
        n=64, initial_count=8, batch_size=8, train_steps=20,
        update_steps=4, replay_size=8, calibration_steps=4,
    )
    return demo["run_demo"](config)


def test_all_six_mlp_runs_reconstruct_their_public_code_lengths(report):
    estimators = {
        "blockwise": BlockwisePrequential,
        "online": OnlinePrequential,
        "replay": ReplayPrequential,
    }
    assert {(r["protocol"], r["calibrated"]) for r in report["rows"]} == {
        (protocol, calibrated)
        for protocol in estimators
        for calibrated in (False, True)
    }
    # The persisted message is sufficient to recompute every score without training.
    persisted = json.loads(json.dumps(report, allow_nan=False))
    for row in persisted["rows"]:
        source = row["ledger"]
        entries = tuple(
            PrequentialEntry(
                **{**entry, "replay_indices": tuple(entry["replay_indices"])}
            )
            for entry in source["entries"]
        )
        ledger = PrequentialLedger(
            source["initial_nll"], source["initial_count"], entries,
            source["n"], row["protocol"],
        )
        result = estimators[row["protocol"]](unit="bits").estimate(inputs=ledger)
        assert result.primary.value.item() == pytest.approx(row["total_bits"])
        assert row["total_bits"] == pytest.approx(
            row["initial_bits"] + row["predictive_bits"]
        )
        assert row["total_bits"] / source["n"] == pytest.approx(
            row["bits_per_observation"]
        )
        assert math.isfinite(row["total_bits"]) and row["total_bits"] >= 0
        assert row["initial_bits"] == pytest.approx(source["initial_count"])
        coded_count = source["initial_count"] + sum(e.stop - e.start for e in entries)
        assert coded_count == source["n"]
        for entry in entries:
            assert entry.trained_until <= entry.start
            assert entry.calibrated_until <= entry.start
            assert all(0 <= i < entry.stop for i in entry.replay_indices)
        if row["protocol"] == "replay":
            assert any(entry.replay_indices for entry in entries)
        else:
            assert not any(entry.replay_indices for entry in entries)
        if row["calibrated"]:
            assert any(entry.calibrated_until > 0 for entry in entries)
        else:
            assert all(entry.calibrated_until == 0 for entry in entries)


def test_seed_reproduces_scores_and_ledgers_without_changing_global_rng(demo):
    config = demo["DemoConfig"](
        n=24, initial_count=4, batch_size=4, train_steps=3,
        update_steps=2, replay_size=4, calibration_steps=2,
    )
    rng_before = torch.random.get_rng_state().clone()
    first = demo["run_demo"](config)
    second = demo["run_demo"](config)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for a, b in zip(first["rows"], second["rows"]):
        assert a["total_bits"] == b["total_bits"]
        assert a["ledger"] == b["ledger"]


def test_zero_replay_has_the_same_learning_trajectory_as_online(demo):
    config = demo["DemoConfig"](
        n=24, initial_count=4, batch_size=4, train_steps=3,
        update_steps=2, replay_size=0, calibration_steps=2,
    )
    rows = demo["run_demo"](config)["rows"]
    for calibrated in (False, True):
        online = next(
            r for r in rows
            if r["protocol"] == "online" and r["calibrated"] == calibrated
        )
        replay = next(
            r for r in rows
            if r["protocol"] == "replay" and r["calibrated"] == calibrated
        )
        assert online["total_bits"] == replay["total_bits"]
        assert online["ledger"] == replay["ledger"]
