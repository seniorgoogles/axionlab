#!/usr/bin/env bash
# Sweep: train (progressive QAT) + export the JSC net for several bit-width targets.
# Each sweep point trains, then immediately exports its own FINN bundle, then the
# next point starts. One run failing (e.g. rolled back below --min-acc) does NOT
# stop the sweep -- it's logged and the sweep continues.
#
# Usage:  ./run_sweep.sh
# Results: sweep_runs/<name>/  (one bundle per point) + sweep_runs/summary.txt
set -uo pipefail
cd "$(dirname "$0")"

# ---- fixed settings ----------------------------------------------------------
EXP=configs/experiments/jsc/jsc_2l_qat.yaml
MODEL=configs/models/jsc_2l.yaml
WEIGHTS=outputs/jsc_2l_qat/best.pth      # where qat writes best.pth (from the exp yaml)
DEVICE=cuda                              # cuda | mps | cpu
WARMUP=15                                # FULL-PRECISION (float) start epochs, same for every point
MIN_ACC=0.7                              # roll back + stop a point if a stage is below this
INPUT_SHAPE=16
INPUT_QUANT=8
SYNTH=ip                                 # estimate | ip | bitfile

# ---- sweep points:  name | schedule | quantize(final stage) ------------------
# format per line:  <folder name> | <wWaA:epochs,...> | <final bit width>
#   name     -> output folder sweep_runs/<name>/
#   schedule -> QAT stages run in order; WARMUP float epochs run first
#   quantize -> must equal the LAST stage (used by export to rebuild the net)
# Each point starts from full precision (WARMUP) and steps DOWN to its target,
# so it's a complete progressive run per bit width.  Edit/add/remove lines freely.
SWEEP=(
  "jsc_w10a10|w10a10:10|w10a10"
  "jsc_w8a8|w10a10:8,w8a8:8|w8a8"
  "jsc_w6a6|w10a10:8,w8a8:6,w6a6:8|w6a6"
  "jsc_w4a4|w10a10:8,w8a8:6,w6a6:6,w4a4:8|w4a4"
  "jsc_w2a2|w10a10:8,w8a8:6,w6a6:6,w4a4:6,w2a2:10|w2a2"
)
# -----------------------------------------------------------------------------

ROOT=sweep_runs
mkdir -p "$ROOT"
SUMMARY="$ROOT/summary.txt"
: > "$SUMMARY"
DB="$ROOT/experiments.db"               # shared DB -> compare all points in one place

for entry in "${SWEEP[@]}"; do
    IFS='|' read -r NAME SCHEDULE QUANTIZE <<< "$entry"
    BUNDLE="$ROOT/$NAME"
    mkdir -p "$BUNDLE"
    LOG="$BUNDLE/train.log"
    echo
    echo "================ sweep point: $NAME  (schedule=$SCHEDULE) ================"

    # tee the run so we can pull the final accuracy out of it afterwards
    if ! python run.py qat --exp "$EXP" \
            --warmup-epochs "$WARMUP" --schedule "$SCHEDULE" --min-acc "$MIN_ACC" \
            --device "$DEVICE" \
            --track-db "$DB" 2>&1 | tee "$LOG"; then
        echo "$NAME  TRAIN_FAILED" | tee -a "$SUMMARY"
        continue
    fi

    # accuracy.txt in the folder: the final eval dict + the per-stage schedule result
    {
        echo "$NAME"
        grep -E '^eval:' "$LOG" | tail -1
        grep -E '^schedule result:' "$LOG" | tail -1
    } > "$BUNDLE/accuracy.txt"
    ACC=$(grep -E '^eval:' "$LOG" | tail -1)

    if ! python run.py export --model "$MODEL" \
            --weights "$WEIGHTS" \
            --quantize "$QUANTIZE" --input-shape "$INPUT_SHAPE" --input-quant "$INPUT_QUANT" \
            --out "$BUNDLE/jsc.qonnx.onnx" --bundle "$BUNDLE" --synth "$SYNTH"; then
        echo "$NAME  EXPORT_FAILED  ($ACC)" | tee -a "$SUMMARY"
        continue
    fi

    echo "$NAME  OK  ($ACC) -> $BUNDLE/" | tee -a "$SUMMARY"
done

echo
echo "==> sweep done. summary:"
cat "$SUMMARY"
echo "==> bundles in $ROOT/<name>/ ; run a FINN build per bundle with:  cd $ROOT/<name> && ./run-docker.sh python build.py"
