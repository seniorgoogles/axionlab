"""Spec for CheckpointManager best/last persistence (TDD for task #12).

Pins down the bug where save_best/save_last deleted the file they had just
written, and that 'best' only updates on genuine improvement.
"""

import torch
import torch.nn as nn

from src.training.checkpoint_manager import CheckpointManager


def _model():
    return nn.Linear(4, 2)


def test_save_last_persists(tmp_path):
    cm = CheckpointManager(str(tmp_path), "m")
    cm.save_last(_model(), epoch=3)
    assert (tmp_path / "last.pth").exists()


def test_save_best_persists(tmp_path):
    cm = CheckpointManager(str(tmp_path), "m")
    saved = cm.save_best(_model(), metric_value=0.5, metric_name="accuracy", mode="max", epoch=1)
    assert saved is True
    assert (tmp_path / "best.pth").exists()


def test_save_best_only_on_improvement_max(tmp_path):
    cm = CheckpointManager(str(tmp_path), "m")
    assert cm.save_best(_model(), 0.5, "accuracy", "max", epoch=1) is True
    assert cm.save_best(_model(), 0.4, "accuracy", "max", epoch=2) is False   # worse -> keep
    assert cm.save_best(_model(), 0.6, "accuracy", "max", epoch=3) is True    # better -> update
    assert (tmp_path / "best.pth").exists()


def test_save_best_only_on_improvement_min(tmp_path):
    cm = CheckpointManager(str(tmp_path), "m")
    assert cm.save_best(_model(), 1.0, "loss", "min", epoch=1) is True
    assert cm.save_best(_model(), 1.5, "loss", "min", epoch=2) is False
    assert cm.save_best(_model(), 0.7, "loss", "min", epoch=3) is True


def test_loadable_state_dict(tmp_path):
    cm = CheckpointManager(str(tmp_path), "m")
    m = _model()
    cm.save_best(m, 0.9, "accuracy", "max", epoch=1)
    reloaded = _model()
    reloaded.load_state_dict(torch.load(tmp_path / "best.pth"))
    for a, b in zip(m.state_dict().values(), reloaded.state_dict().values()):
        assert torch.allclose(a, b)
