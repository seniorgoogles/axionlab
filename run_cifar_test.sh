#!/usr/bin/env bash
# Quick toolchain test for ResNet-18 on CIFAR-10 (NOT a sweep).
# Goal: just check the chain works -- train a couple of quantized epochs, then
# export to QONNX for the Kria part. Accuracy is irrelevant here.
# Usage:  ./run_cifar_test.sh
set -euo pipefail
cd "$(dirname "$0")"

EXP=configs/experiments/cifar/resnet18_cifar.yaml
MODEL=configs/models/resnet18_cifar.yaml
WEIGHTS=outputs/resnet18_cifar/best.pth
BUNDLE=cifar_test                         # export bundle goes here
DEVICE=cuda                               # cuda | mps | cpu
FPGA_PART=xck26-sfvc784-2LV-c            # Kria K26 SoM

echo "==> [1/2] short QAT (2 quantized epochs, no warmup, no rollback)"
python run.py qat --exp "$EXP" \
    --warmup-epochs 0 --schedule w4a4:2 \
    --device "$DEVICE" \
    --track-db "$BUNDLE/experiments.db"

echo "==> [2/2] export -> $BUNDLE/ (synth estimate, Kria part)"
python run.py export --model "$MODEL" \
    --weights "$WEIGHTS" \
    --quantize w4a4 --input-shape 3,32,32 --input-quant 8 \
    --fpga-part "$FPGA_PART" \
    --out "$BUNDLE/model.qonnx.onnx" --bundle "$BUNDLE" --synth estimate

echo "==> export done. now test the FINN build (estimate):"
echo "    cd $BUNDLE && ./run-docker.sh python build.py"
