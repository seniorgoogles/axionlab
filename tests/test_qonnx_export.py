"""Spec for QONNX export (task #14).

Needs brevitas + onnx + onnxoptimizer (brevitas writes ONNX through them). The
qonnx cleanup pass is optional. Skipped if any of these aren't installed.
"""

import torch
import torch.nn as nn
import pytest

pytest.importorskip("brevitas")
pytest.importorskip("onnx")
pytest.importorskip("onnxoptimizer")


def _materialize(model, *shape):
    """Run one forward so brevitas creates its lazy activation-scale params."""
    model.train()
    with torch.no_grad():
        model(torch.zeros(1, *shape))


def test_add_input_quant_forward():
    from src.export import add_input_quant
    model = nn.Conv2d(3, 4, 3, padding=1)
    wrapped = add_input_quant(model, bit_width=8)
    y = wrapped(torch.randn(1, 3, 8, 8))
    assert y.shape == (1, 4, 8, 8)


def test_export_to_qonnx_writes_file(tmp_path):
    import brevitas.nn as qnn
    from src.export import add_input_quant, export_to_qonnx

    model = nn.Sequential(qnn.QuantConv2d(3, 4, 3, padding=1, weight_bit_width=4),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    model = add_input_quant(model, bit_width=8)
    out = tmp_path / "m.qonnx.onnx"
    export_to_qonnx(model, input_shape=(3, 16, 16), path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_surgery_then_export_roundtrip(tmp_path):
    """The real path: float yaml -> set_bit_width (QAT surgery) -> save weights,
    then rebuild + set_bit_width + load (like `export --quantize w4a4`) -> QONNX.
    A dummy forward materializes the lazy quant params so the strict load matches."""
    from src.core.build import build_model
    from src.core.quantize import set_bit_width
    from src.export import add_input_quant, export_to_qonnx

    cfg = {"nc": 2, "backbone": [
        [-1, 1, "Linear", "fc1", {"in_features": 16, "out_features": 8}],
        [-1, 1, "ReLU", "relu1", {}],
        [-1, 1, "Linear", "fc2", {"in_features": 8, "out_features": 2}],
    ]}

    trained = set_bit_width(build_model(cfg), 4, 4)
    _materialize(trained, 16)
    ckpt = tmp_path / "best.pth"
    torch.save(trained.state_dict(), ckpt)

    exported = set_bit_width(build_model(cfg), 4, 4)
    _materialize(exported, 16)                       # match the trained checkpoint's keys
    exported.load_state_dict(torch.load(ckpt))       # strict load must succeed
    exported = add_input_quant(exported, bit_width=8)

    out = tmp_path / "jsc.qonnx.onnx"
    export_to_qonnx(exported, input_shape=(16,), path=str(out))
    assert out.exists() and out.stat().st_size > 0
