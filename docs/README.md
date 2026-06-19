# axionlab docs

axionlab builds, trains, quantizes and deploys quantized neural networks. It glues
together **Ultralytics** (model zoo), **Brevitas** (quantization-aware training),
**QONNX/FINN** (FPGA dataflow) behind one config + CLI:

```
ultralytics / yaml  →  Brevitas QAT  →  QONNX  →  FINN  →  FPGA
```

## Pages

- [CLI reference](cli.md) — `train` / `qat` / `infer` / `eval`
- [Configs](configs.md) — experiment yaml + model (net) yaml format
- [Training](training.md) — the Runner, phases, callbacks, checkpoints, tracking DB
- [Quantization & QAT](qat.md) — quant layers, `quantize_model`, progressive schedules, QONNX/FINN export
- [Datasets](datasets.md) — the dataset factory, built-ins, custom preprocessing
- [Inference](inference.md) — predictor + GUI
- [Architecture & cleanup](../CLEANUP.md) — package layout, design decisions, dead-code list

## Install

Core needs `torch`. Optional features pull extra deps:

| feature | package |
| --- | --- |
| QAT / quant layers | `brevitas` |
| QONNX export | `brevitas`, `onnx`, `onnxoptimizer` (qonnx cleanup optional) |
| vision datasets (CIFAR/MNIST/ImageFolder) | `torchvision` |
| OpenML / JSC dataset | `scikit-learn` |
| live dashboard | `rich` |
| progress bars | `tqdm` |
| YOLO graph builder | `ultralytics` |
| FPGA build | FINN (inside its Docker image) |

```sh
pip install torch brevitas qonnx torchvision scikit-learn rich tqdm
```

## Quickstart

```sh
# run the test suite
python -m pytest tests/ -q

# quick pipeline smoke test (1 epoch, few batches)
python run.py train --exp configs/experiments/jsc/jsc_2l_qat.yaml --smoke

# progressive QAT with a live dashboard
python run.py qat --exp configs/experiments/jsc/jsc_2l_qat.yaml \
    --warmup-epochs 5 --schedule w8a8:6,w4a4:6,w2a2:8 --min-acc 0.7 --dashboard
```

## Layout

```
src/
  core/        builders (sequential + graph), Runner, quantize, qat_schedule
  datasets/    factory + built-in datasets + preprocessing
  training/    TrainerConfig, CheckpointManager (Runner is the entry point)
  export/      QONNX export
  finn/        FINN dataflow build wrapper
  inference/   backend-agnostic predictors
  gui/         Tkinter inference GUI
  ui/          Rich training dashboard
  tracking/    SQLite experiment DB
  cli.py       command-line interface
configs/
  models/      net definitions (resnet18, yolov8n, jsc_2l, ...)
  experiments/ experiment definitions (model + dataset + training)
examples/      runnable scripts
tests/         pytest suite
```
