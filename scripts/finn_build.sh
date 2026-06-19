#!/usr/bin/env sh
# One-shot FINN build: copy the QONNX file into the FINN repo, generate the build
# script, and run it inside the FINN Docker container — all from one command.
#
#   sh scripts/finn_build.sh [path/to/model.qonnx.onnx]
#
# Env overrides:
#   FINN_ROOT=/path/to/finn      (default: ~/Documents/repositories/finn)
#   FPGA_PART=xc7z020clg400-1    OUTPUT=jsc_finn_build    CLK_NS=10.0
set -e

QONNX="${1:-jsc.qonnx.onnx}"
FINN_ROOT="${FINN_ROOT:-$HOME/Documents/repositories/finn}"
FPGA_PART="${FPGA_PART:-xc7z020clg400-1}"
OUTPUT="${OUTPUT:-jsc_finn_build}"
CLK_NS="${CLK_NS:-10.0}"

[ -f "$QONNX" ] || { echo "QONNX file not found: $QONNX (run 'python run.py export ...' first)"; exit 1; }
[ -d "$FINN_ROOT" ] || { echo "FINN repo not found at $FINN_ROOT (set FINN_ROOT=...)"; exit 1; }

# 1) copy the model into the FINN repo (mounted into the container)
cp "$QONNX" "$FINN_ROOT/_axionlab_model.qonnx.onnx"
echo "copied $QONNX -> $FINN_ROOT/_axionlab_model.qonnx.onnx"

# 2) generate the in-container build script (estimate reports; no Vivado needed)
cat > "$FINN_ROOT/_axionlab_build.py" <<PY
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as cfg

build_cfg = cfg.DataflowBuildConfig(
    output_dir="$OUTPUT",
    synth_clk_period_ns=$CLK_NS,
    fpga_part="$FPGA_PART",
    generate_outputs=[
        cfg.DataflowOutputType.ESTIMATE_REPORTS,
        # uncomment for real synthesis (needs Vivado in the container):
        # cfg.DataflowOutputType.STITCHED_IP,
        # cfg.DataflowOutputType.BITFILE,
    ],
)
build.build_dataflow_cfg("_axionlab_model.qonnx.onnx", build_cfg)
print("done -> $OUTPUT/ (report/*.json, intermediate_models/*.onnx)")
PY
echo "wrote $FINN_ROOT/_axionlab_build.py"

# 3) start the FINN container and run the build non-interactively
cd "$FINN_ROOT"
echo "running FINN build in the Docker container…"
./run-docker.sh python _axionlab_build.py
