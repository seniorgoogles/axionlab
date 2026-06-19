"""Spec for the progressive QAT schedule with rollback (tasks: bit-width schedule)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("brevitas")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core import Runner
from src.core.qat_schedule import QatStage, run_qat_schedule
from src.training.trainer_config import TrainerConfig


def _runner(tmp_path):
    x = torch.randn(40, 8)
    y = torch.randint(0, 3, (40,))
    loader = DataLoader(TensorDataset(x, y), batch_size=10)
    cfg = TrainerConfig(save_dir=str(tmp_path), model_name="t", epochs=1, lr=0.01, validate_every=1)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    r = Runner(model, cfg, loader, loader, device="cpu")
    r.show_progress = False
    return r


def test_schedule_lowers_bit_width_through_stages(tmp_path):
    import brevitas.nn as qnn

    runner = _runner(tmp_path)
    run_qat_schedule(runner, [
        QatStage("w8a8", weight_bit_width=8, act_bit_width=8, epochs=1, lr=1e-3),
        QatStage("w4a4", weight_bit_width=4, act_bit_width=4, epochs=1, lr=1e-3),
    ])
    # model is quantized after the schedule
    assert any(isinstance(m, qnn.QuantLinear) for m in runner.model.modules())


def test_schedule_rolls_back_on_missed_accuracy(tmp_path):
    runner = _runner(tmp_path)
    result = run_qat_schedule(runner, [
        QatStage("w8a8", weight_bit_width=8, act_bit_width=8, epochs=1, lr=1e-3),
        # impossible floor -> this stage must roll back and stop
        QatStage("w2a2", weight_bit_width=2, act_bit_width=2, epochs=1, lr=1e-3, min_accuracy=2.0),
    ])
    assert result["stopped_at"] == "w2a2"
    assert result["rolled_back"] is True
