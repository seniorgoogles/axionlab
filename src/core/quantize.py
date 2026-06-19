"""Float -> quantized model surgery for QAT.

- ``quantize_model(model, wb, ab)``: float -> Brevitas quant, skipping layers that
  are already quantized (idempotent). Use to enter QAT once.
- ``set_bit_width(model, wb, ab)``: (re)create quant layers at a target bit width,
  also lowering existing quant layers. Use for a PROGRESSIVE schedule, e.g.
  w8a8 -> w4a4 -> w2a2 across phases.

Both copy weights so training continues rather than restarting. Used as a
``Phase.on_start`` hook via ``runner.replace_model(...)``.
"""

from __future__ import annotations

import torch.nn as nn


def _requantize(model: nn.Module, weight_bit_width: int, act_bit_width: int,
                only_float: bool) -> nn.Module:
    """Walk the model and (re)build conv/linear/relu as Brevitas quant layers.

    only_float=True  -> skip layers that are already quantized (entry into QAT).
    only_float=False -> rebuild every conv/linear/relu at the given width
                        (lowering existing quant layers too).
    """
    import brevitas.nn as qnn

    counts = {"conv": 0, "linear": 0, "act": 0}

    def visit(module):
        # Replace conv/linear/relu LEAVES; never recurse INTO a quant layer
        # (its internal proxies are not user layers and must not be touched).
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                if not (only_float and isinstance(child, qnn.QuantConv2d)):
                    q = qnn.QuantConv2d(
                        child.in_channels, child.out_channels, child.kernel_size,
                        stride=child.stride, padding=child.padding, dilation=child.dilation,
                        groups=child.groups, bias=child.bias is not None,
                        weight_bit_width=weight_bit_width,
                    )
                    q.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        q.bias.data.copy_(child.bias.data)
                    setattr(module, name, q)
                    counts["conv"] += 1
            elif isinstance(child, nn.Linear):
                if not (only_float and isinstance(child, qnn.QuantLinear)):
                    q = qnn.QuantLinear(
                        child.in_features, child.out_features,
                        bias=child.bias is not None, weight_bit_width=weight_bit_width,
                    )
                    q.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        q.bias.data.copy_(child.bias.data)
                    setattr(module, name, q)
                    counts["linear"] += 1
            elif isinstance(child, (nn.ReLU, nn.SiLU)) or isinstance(child, qnn.QuantReLU):
                if not (only_float and isinstance(child, qnn.QuantReLU)):
                    setattr(module, name, qnn.QuantReLU(bit_width=act_bit_width))
                    counts["act"] += 1
            else:
                visit(child)  # recurse only into plain containers

    visit(model)
    print(f"[quantize] {counts['conv']} conv + {counts['linear']} linear + "
          f"{counts['act']} act -> Brevitas (Wb={weight_bit_width}, Ab={act_bit_width})")
    return model


def quantize_model(model: nn.Module, weight_bit_width: int = 4, act_bit_width: int = 4) -> nn.Module:
    """Float -> quant (idempotent: skips already-quantized layers)."""
    return _requantize(model, weight_bit_width, act_bit_width, only_float=True)


def set_bit_width(model: nn.Module, weight_bit_width: int, act_bit_width: int) -> nn.Module:
    """(Re)quantize all conv/linear/relu to the target width, lowering existing
    quant layers too. For progressive QAT schedules."""
    return _requantize(model, weight_bit_width, act_bit_width, only_float=False)
