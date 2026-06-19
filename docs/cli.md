# CLI reference

Entry point: `python run.py <command> [flags]` (or `python -m src.cli ...`).
Help is hierarchical — the built-in `-h` is the source of truth:

```sh
python run.py -h            # lists the commands
python run.py train -h      # flags + examples for `train`
python run.py qat -h
python run.py infer -h
python run.py eval -h
```

CLI flags **override** the experiment yaml (the yaml is the base).

## Common flags

| flag | meaning |
| --- | --- |
| `--exp / --experiment YAML` | experiment yaml (required) |
| `--device cuda\|mps\|cpu` | compute device (default: auto CUDA/ROCm → MPS → CPU) |

## `train`

Train a model from an experiment yaml.

| flag | meaning |
| --- | --- |
| `--epochs N` | override epoch count |
| `--lr X` | override learning rate |
| `--track-db PATH` | record metrics + weights + yamls into a SQLite DB |
| `--early-stop plateau` + `--patience N` | stop when the best metric plateaus |
| `--dashboard` | live Rich dashboard |
| `--dry-run` | train but write no checkpoints |
| `--smoke` | tiny fast run (1 epoch, few batches) |

```sh
python run.py train --exp configs/experiments/jsc/jsc_2l_qat.yaml \
    --epochs 10 --lr 0.03 --device cuda --dashboard \
    --early-stop plateau --patience 5 --dry-run

python run.py train --exp configs/experiments/jsc/jsc_2l_qat.yaml --smoke
```

## `qat`

Progressive quantization-aware training with rollback. Optional float warmup, then
lower the bit width in stages; a stage that misses `--min-acc` rolls back to the
previous width and stops. Weights and activations are scheduled independently.

| flag | meaning |
| --- | --- |
| `--warmup-epochs N` | float epochs before quantizing |
| `--schedule wWaA:E,...` | bit-width stages (W=weight bits, A=act bits, E=epochs) |
| `--lr X` | per-stage learning rate |
| `--min-acc X` | roll back + stop if a stage's best metric < X |
| `--track-db PATH`, `--dashboard` | as in `train` |

```sh
python run.py qat --exp configs/experiments/jsc/jsc_2l_qat.yaml \
    --warmup-epochs 5 --schedule w8a8:6,w4a4:6,w2a2:8 --min-acc 0.7 --dashboard

# lower weights first, then activations:
python run.py qat --exp E.yaml --schedule w8a8:6,w4a8:6,w4a4:6
```

## `infer`

Load a checkpoint and run inference.

| flag | meaning |
| --- | --- |
| `--weights PTH` | checkpoint to load |
| `--source SRC` | `webcam` \| an image path \| `dataset` |
| `--gui` | open the interactive GUI (iterate dataset / webcam / screenshots) |
| `--save-dir DIR` | where the GUI saves screenshots |

```sh
python run.py infer --exp configs/models/resnet18.yaml --weights best.pth --source webcam --gui
python run.py infer --exp configs/models/resnet18.yaml --weights best.pth --source bild.jpg
```

## `eval`

```sh
python run.py eval --exp E.yaml --weights best.pth
```

## `export`

Build the net from its yaml, load weights, optionally quantize the input, and
export QONNX for FINN.

| flag | meaning |
| --- | --- |
| `--model YAML` | net yaml (the architecture) |
| `--weights PTH` | checkpoint to load |
| `--input-shape C,H,W` | input shape without batch (e.g. `3,224,224` or `16`) |
| `--input-quant BITS` | wrap the input in a QuantIdentity of this bit width |
| `--quantize wWaA` | rebuild the net as quant before loading (needed when the checkpoint came from quantize-and-finetune surgery) |
| `--out PATH` | output `.onnx` (default `model.qonnx.onnx`) |

```sh
# checkpoint trained via the surgery flow (float yaml + QAT phases) -> reconstruct quant arch:
python run.py export --model configs/models/jsc_2l.yaml --weights outputs/jsc_2l_qat/best.pth \
    --quantize w4a4 --input-shape 16 --input-quant 8 --out jsc.qonnx.onnx
```

`--quantize` matches the bit width the checkpoint was trained at (e.g. after a
schedule that rolled back to w4a4, use `--quantize w4a4`). Skip it if the model
yaml already defines quant layers.

## `finn`

Run the FINN dataflow build on a QONNX model (inside the FINN Docker image).

| flag | meaning |
| --- | --- |
| `qonnx` (positional) | QONNX model path (from `export`) |
| `--output DIR` | FINN build output dir |
| `--fpga-part PART` | target FPGA part |
| `--clk-ns X` | target clock period (ns) |
| `--no-rtlsim` | estimates only (skip stitched-IP / rtlsim) |

```sh
python run.py finn jsc.qonnx.onnx --fpga-part xc7z020clg400-1 --output build
```

Full chain on the command line:

```sh
python run.py qat    --exp .../jsc_2l_qat.yaml --schedule w8a8:6,w4a4:6 --min-acc 0.7
python run.py export --model configs/models/jsc_2l.yaml --weights outputs/.../best.pth \
    --quantize w4a4 --input-shape 16 --input-quant 8 --out jsc.qonnx.onnx
python run.py finn   jsc.qonnx.onnx --fpga-part xc7z020clg400-1   # inside FINN Docker
```
