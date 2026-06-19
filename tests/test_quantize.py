"""Spec for the float->quant swap used by the QAT phase recipe."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("brevitas")

import torch
import torch.nn as nn

import brevitas.nn as qnn

from src.core.quantize import quantize_model


def _mlp():
    return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 5))


def test_swaps_linear_and_relu_and_copies_weights():
    model = _mlp()
    w0 = model[0].weight.detach().clone()
    quantize_model(model, weight_bit_width=4, act_bit_width=4)

    assert isinstance(model[0], qnn.QuantLinear)
    assert isinstance(model[1], qnn.QuantReLU)
    assert isinstance(model[2], qnn.QuantLinear)
    # weights carried over from the float model (so QAT fine-tunes, not restarts)
    assert torch.allclose(model[0].weight.detach(), w0)


def test_quantized_model_still_forwards():
    model = quantize_model(_mlp(), 4, 4)
    y = model(torch.randn(3, 16))
    assert y.shape == (3, 5)


def test_idempotent_no_double_wrap():
    model = quantize_model(_mlp(), 4, 4)
    quantize_model(model, 4, 4)  # second pass must not re-wrap Quant layers
    assert isinstance(model[0], qnn.QuantLinear)


def test_set_bit_width_requantizes_existing():
    from src.core.quantize import set_bit_width

    model = _mlp()
    set_bit_width(model, 8, 8)
    assert isinstance(model[0], qnn.QuantLinear)
    first = model[0]
    set_bit_width(model, 2, 2)               # lower an already-quant layer
    assert isinstance(model[0], qnn.QuantLinear)
    assert model[0] is not first             # re-created at the lower width
    assert model(torch.randn(2, 16)).shape == (2, 5)
