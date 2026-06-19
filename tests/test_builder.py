"""Spec for the sequential model builder (task #4 / validation #15)."""

import pytest

pytest.importorskip("torch")

import torch

from src.core.build import build_model, load_float_state_dict


def _mlp_config():
    return {
        "nc": 2,
        "backbone": [
            [-1, 1, "Linear", "fc1", {"in_features": 4, "out_features": 8}],
            [-1, 1, "ReLU", "act", {}],
            [-1, 1, "Linear", "fc2", {"in_features": 8, "out_features": 2}],
        ],
    }


def test_build_model_forward_shape():
    model = build_model(_mlp_config())
    y = model(torch.randn(3, 4))
    assert y.shape == (3, 2)


def test_named_modules_match_config():
    model = build_model(_mlp_config())
    names = dict(model.named_modules())
    assert "fc1" in names and "fc2" in names and "act" in names


def test_load_float_state_dict_by_name():
    src = build_model(_mlp_config())
    dst = build_model(_mlp_config())
    load_float_state_dict(dst, src, verbose=False)
    for a, b in zip(src.state_dict().values(), dst.state_dict().values()):
        assert torch.allclose(a, b)
