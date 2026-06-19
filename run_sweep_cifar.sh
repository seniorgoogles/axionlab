#!/usr/bin/env bash
# Sweep: train (progressive QAT) + export ResNet-18 on CIFAR-10 for several bit
# widths, targeting the Kria K26 SoM (KV260/KR260, part xck26-sfvc784-2LV-c).
# Each point trains, then immediately exports its own FINN bundle; a failed point
# is logged and the sweep continues.
#
# Usage:  ./run_sweep_cifar.sh
# Results: sweep_runs_cifar/<name>/  + sweep_runs_cifar/summary.txt
set -uo pipefail
cd "$(dirname "$0")"

# ---- fixed settings ----------------------------------------------------------
EXP=configs/experiments/cifar/resnet18_cifar.yaml
MODEL=configs/models/resnet18_cifar.yaml
WEIGHTS=outputs/resnet18_cifar/best.pth  # where qat writes best.pth (from the exp yaml)
DEVICE=cuda                              # cuda | mps | cpu
WARMUP=30                                # FULL-PRECISION (float) start epochs (ResNet needs more)
MIN_ACC=0.5                              # roll back + stop a point below this (lenient for low bits)
INPUT_SHAPE=3,32,32                      # CIFAR image C,H,W
INPUT_QUANT=8
FPGA_PART=xck26-sfvc784-2LV-c           # Kria K26 SoM (KV260/KR260)
SYNTH=estimate                          # estimate first (ResNet is big); switch to ip/bitfile later

# ---- sweep points:  name | schedule | quantize(final stage) ------------------
# lean 3-point curve: near-lossless / sweet-spot / aggressive. Edit freely.
SWEEP=(
  "resnet18_cifar_w8a8|w8a8:20|w8a8"
  "resnet18_cifar_w4a4|w8a8:15,w4a4:20|w4a4"
  "resnet18_cifar_w2a2|w8a8:12,w4a4:12,w2a2:20|w2a2"
)
# -----------------------------------------------------------------------------

ROOT=sweep_runs_cifar
mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"
: > "$SUMMARY"
DB="$ROOT/experiments.db"

for entry in "${SWEEP[@]}"; do
    IFS='|' read -r NAME SCHEDULE QUANTIZE <<< "$entry"
    BUNDLE="$ROOT/$NAME"
    mkdir -p "$BUNDLE"
    LOG="$BUNDLE/train.log"
    echo
    echo "================ sweep point: $NAME  (schedule=$SCHEDULE) ================"

    if ! python run.py qat --exp "$EXP" \
            --warmup-epochs "$WARMUP" --schedule "$SCHEDULE" --min-acc "$MIN_ACC" \
            --device "$DEVICE" \
            --track-db "$DB" 2>&1 | tee "$LOG"; then
        echo "$NAME  TRAIN_FAILED" | tee -a "$SUMMARY"
        continue
    fi

    {
        echo "$NAME"
        grep -E '^eval:' "$LOG" | tail -1
        grep -E '^schedule result:' "$LOG" | tail -1
    } > "$BUNDLE/accuracy.txt"
    ACC=$(grep -E '^eval:' "$LOG" | tail -1)

    if ! python run.py export --model "$MODEL" \
            --weights "$WEIGHTS" \
            --quantize "$QUANTIZE" --input-shape "$INPUT_SHAPE" --input-quant "$INPUT_QUANT" \
            --fpga-part "$FPGA_PART" \
            --out "$BUNDLE/model.qonnx.onnx" --bundle "$BUNDLE" --synth "$SYNTH"; then
        echo "$NAME  EXPORT_FAILED  ($ACC)" | tee -a "$SUMMARY"
        continue
    fi

    echo "$NAME  OK  ($ACC) -> $BUNDLE/" | tee -a "$SUMMARY"
done

echo
echo "==> sweep done. summary:"
cat "$SUMMARY"
echo "==> bundles in $ROOT/<name>/ ; run a FINN build per bundle with:  cd $ROOT/<name> && ./run-docker.sh python build.py"
