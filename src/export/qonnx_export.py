"""Export a (quantized) model to QONNX so FINN can ingest it.

FINN expects the network input to be quantized too. `add_input_quant` wraps a
model with a uint8 QuantIdentity at the entry; `export_to_qonnx` runs Brevitas'
QONNX export and the QONNX cleanup pass.

    from src.export import export_to_qonnx, add_input_quant
    model = add_input_quant(model, bit_width=8)          # quantize the input
    export_to_qonnx(model, input_shape=(3, 224, 224), path="model.qonnx.onnx")

The resulting file feeds the FINN build flow (see src/finn/build_finn.py, task #18).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantInputWrapper(nn.Module):
    """Quantize the raw input once, then run the wrapped model.

    Kept as a thin wrapper (single responsibility) so the underlying model stays
    untouched and re-usable for the float path.
    """

    def __init__(self, model: nn.Module, bit_width: int = 8):
        super().__init__()
        import brevitas.nn as qnn
        from brevitas.quant import Int8ActPerTensorFloat

        # SIGNED input quant: FINN requires identity Quant nodes (the network input,
        # with no preceding activation) to be signed. Also correct for inputs that
        # take negative values (e.g. standardized tabular features).
        self.input_quant = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat, bit_width=bit_width, return_quant_tensor=False)
        self.model = model

    def forward(self, x):
        return self.model(self.input_quant(x))


def add_input_quant(model: nn.Module, bit_width: int = 8) -> QuantInputWrapper:
    return QuantInputWrapper(model, bit_width=bit_width)


def export_to_qonnx(model: nn.Module, input_shape, path: str, batch: int = 1) -> str:
    """Export `model` to QONNX at `path`.

    Args:
        model: a Brevitas-quantized model (ideally input already quantized).
        input_shape: shape WITHOUT batch dim, e.g. (3, 224, 224) or (16,).
        path: output .onnx path.
    Returns the path written.
    """
    from brevitas.export import export_qonnx

    model.eval()
    dummy = torch.randn(batch, *input_shape)

    # torch 2.x defaults to the dynamo ONNX exporter, which has no symbolic for
    # brevitas' custom `qonnx.int_quant` op. Force the legacy TorchScript exporter
    # (brevitas registers its ONNX symbolics there). Fall back if the kwarg is
    # unknown (older torch already uses the legacy path).
    try:
        export_qonnx(model, dummy, path, dynamo=False)
    except TypeError:
        export_qonnx(model, dummy, path)

    # cleanup is a nice-to-have tidy pass; brevitas already wrote valid QONNX.
    # Skip it gracefully if the standalone qonnx tooling isn't importable.
    try:
        from qonnx.util.cleanup import cleanup as qonnx_cleanup
        qonnx_cleanup(path, out_file=path)
    except Exception as e:  # noqa: BLE001
        print(f"[export] qonnx cleanup skipped ({type(e).__name__}); raw QONNX written to {path}")
    return path
