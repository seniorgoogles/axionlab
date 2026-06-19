#!/usr/bin/env bash
# Train (progressive QAT) + export the JSC 2-layer net for FINN, in one go.
# Usage:  ./run_jsc.sh
# Stops immediately if the training fails, so a bad run never gets exported.
set -euo pipefail

# run from the repo root (this script's folder), regardless of where it's called
cd "$(dirname "$0")"

# ---- knobs -------------------------------------------------------------------
EXP=configs/experiments/jsc/jsc_2l_qat.yaml
MODEL=configs/models/jsc_2l.yaml
OUTDIR=outputs/jsc_2l_qat                 # best.pth + experiments.db land here
BUNDLE=finn_jsc                           # export bundle (model, build.py, weights)

WARMUP=15                                 # float warmup epochs
SCHEDULE=w8a8:10,w4a4:8                   # QAT stages: weight/act bits : epochs
MIN_ACC=0.7                               # roll back + stop if a stage is below this
DEVICE=cuda                               # cuda | mps | cpu  (or drop --device to auto-detect)

QUANTIZE=w4a4                             # must match the final QAT stage
INPUT_SHAPE=16                            # JSC = 16 features
INPUT_QUANT=8                             # input quantized to 8 bit
SYNTH=ip                                  # estimate | ip | bitfile
# -----------------------------------------------------------------------------

echo "==> [1/2] QAT training"
python run.py qat --exp "$EXP" \
    --warmup-epochs "$WARMUP" --schedule "$SCHEDULE" --min-acc "$MIN_ACC" \
    --device "$DEVICE" --dashboard \
    --track-db "$OUTDIR/experiments.db"

echo "==> [2/2] export -> $BUNDLE/"
python run.py export --model "$MODEL" \
    --weights "$OUTDIR/best.pth" \
    --quantize "$QUANTIZE" --input-shape "$INPUT_SHAPE" --input-quant "$INPUT_QUANT" \
    --out "$BUNDLE/jsc.qonnx.onnx" --bundle "$BUNDLE" --synth "$SYNTH"

echo "==> done. bundle ready in $BUNDLE/  (run the FINN build with: cd $BUNDLE && ./run-docker.sh python build.py)"
