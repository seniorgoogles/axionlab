"""End-to-end smoke spec for the Runner (task #15). Runs on CPU with torch only."""

import pytest

pytest.importorskip("torch")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core import Phase, Runner
from src.training.trainer_config import TrainerConfig


def _loaders(n=20, d=4, c=3, bs=5):
    x = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    loader = DataLoader(TensorDataset(x, y), batch_size=bs)
    return loader, loader


def _runner(tmp_path, epochs=2, model=None):
    train, val = _loaders()
    cfg = TrainerConfig(save_dir=str(tmp_path), model_name="t", epochs=epochs,
                        lr=0.01, validate_every=1)
    return Runner(model or nn.Linear(4, 3), cfg, train, val, device="cpu")


def test_train_writes_best_last_and_history(tmp_path):
    runner = _runner(tmp_path, epochs=2)
    history = runner.train()
    assert (tmp_path / "best.pth").exists()
    assert (tmp_path / "last.pth").exists()
    assert (tmp_path / "metrics_history.json").exists()
    assert len(history["epoch"]) == 2


def test_evaluate_loads_best(tmp_path):
    runner = _runner(tmp_path, epochs=1)
    runner.train()
    metrics = runner.evaluate()  # loads best.pth
    assert "accuracy" in metrics and "loss" in metrics


def test_plan_runs_all_phases_and_calls_on_start_once(tmp_path):
    runner = _runner(tmp_path, epochs=1)
    calls = {"n": 0}

    def on_start(_runner):
        calls["n"] += 1

    history = runner.train(plan=[
        Phase("warmup", epochs=1, lr=0.01),
        Phase("qat", epochs=2, lr=0.001, on_start=on_start),
    ])
    assert calls["n"] == 1                 # on_start fired exactly once
    assert len(history["epoch"]) == 3      # 1 + 2 epochs
    assert set(history["phase"]) == {"warmup", "qat"}


def test_phase_callbacks_fire(tmp_path):
    runner = _runner(tmp_path, epochs=1)
    seen = {"epochs": 0, "phase_starts": 0}

    class CB:
        def on_phase_start(self, r, phase):
            seen["phase_starts"] += 1

        def on_epoch_end(self, r, epoch, metrics):
            seen["epochs"] += 1

    runner.train(plan=[Phase("a", epochs=2, lr=0.01, callbacks=[CB()])])
    assert seen["phase_starts"] == 1
    assert seen["epochs"] == 2
