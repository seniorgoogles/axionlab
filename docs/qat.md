# Quantization & QAT

axionlab uses **Brevitas** for quantization. There are three ways to get a
quantized model, and a progressive schedule for pushing bit width down.

## 1. Build quantized directly

Use `QuantLinear` / `QuantConv2d` / `QuantReLU` in the model yaml with `bit_width`
(see [configs.md](configs.md)). The Runner then trains it quantization-aware.

## 2. Dedicated quant blocks (FINN-friendly)

`QuantConv`, `QuantC2f`, `QuantBottleneck`, `QuantSPPF` (`src.core.build.quant_blocks`)
put **shared-scale** `QuantIdentity` at every `cat` / residual `add` boundary, which
FINN needs to merge those ops. Use them by name in a graph yaml.

## 3. Float → quantize → fine-tune (surgery)

`src.core.quantize`:

```python
from src.core import quantize_model, set_bit_width
quantize_model(model, weight_bit_width=4, act_bit_width=4)  # float→quant, idempotent
set_bit_width(model, 2, 2)                                  # re-quantize existing layers lower
```

`quantize_model` swaps `Conv2d/Linear → QuantConv2d/QuantLinear` (copying weights so
training continues) and `ReLU/SiLU → QuantReLU`; it skips already-quant layers.
`set_bit_width` rebuilds every quant/float layer at a target width — used for
progressive schedules. Weights and activations take **independent** bit widths.

## Progressive schedule with rollback

`src.core.qat_schedule.run_qat_schedule` lowers the bit width in stages; a stage that
misses its `min_accuracy` rolls the model back to the previous (good) bit width and
stops. Weights and activations are scheduled independently (each stage names both).

```python
from src.core import Runner, Phase
from src.core.qat_schedule import QatStage, run_qat_schedule

runner = Runner.from_experiment("...yaml")
runner.train(plan=[Phase("float_warmup", 8, lr=1e-3)])      # optional warmup
run_qat_schedule(runner, [
    QatStage("w8a8", 8, 8, epochs=6, lr=5e-4, min_accuracy=0.74),
    QatStage("w4a8", 4, 8, epochs=6, lr=3e-4, min_accuracy=0.73),  # weights first
    QatStage("w4a4", 4, 4, epochs=6, lr=2e-4, min_accuracy=0.72),  # then activations
    QatStage("w2a2", 2, 2, epochs=8, lr=1e-4, min_accuracy=0.68),
])
```

Or from the CLI: `python run.py qat --schedule w8a8:6,w4a4:6,w2a2:8 --min-acc 0.7`
(see [cli.md](cli.md)). Example script: `examples/jsc_qat_progressive.py`.

## Export to QONNX / FINN

```python
from src.export import add_input_quant, export_to_qonnx
model = add_input_quant(model, bit_width=8)        # quantize the network input
export_to_qonnx(model, input_shape=(16,), path="model.qonnx.onnx")
```

Then build the FPGA dataflow accelerator (inside the FINN Docker image):

```python
from src.finn import build_finn
build_finn("model.qonnx.onnx", output_dir="build", fpga_part="xc7z020clg400-1")
```

This closes the loop: axionlab/Brevitas QAT → QONNX → FINN → FPGA. The quantized
activations become `MultiThreshold` ops in FINN (see the FloPoCo thermometer-encoder
work for the hardware side).
