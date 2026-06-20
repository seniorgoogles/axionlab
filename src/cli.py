"""axionlab command-line interface.

    python -m src.cli train --exp E.yaml [--epochs N --lr X --device cuda --dashboard
                                          --track-db PATH --early-stop plateau --patience N
                                          --dry-run --smoke]
    python -m src.cli qat   --exp E.yaml [--warmup-epochs N --schedule w8a8:6,w4a4:6,w2a2:8
                                          --min-acc X --lr X --dashboard --track-db PATH]
    python -m src.cli infer --exp E.yaml --weights best.pth --source webcam|<path>|dataset
                                         [--gui --device cuda --save-dir DIR]
    python -m src.cli eval  --exp E.yaml [--weights best.pth --device cuda]

CLI flags override the experiment yaml. Heavy imports (torch/brevitas) are done
lazily inside the handlers so `--help` and arg parsing stay fast and dependency-free.
"""

from __future__ import annotations

import argparse
import re
from typing import List


# ---- pure parsing (stdlib only, unit-tested) ------------------------------

def parse_schedule(spec: str, lr: float, min_acc=None) -> List[dict]:
    """'w8a8:6,w4a4:6' -> list of stage dicts (weight/act bits + epochs)."""
    stages = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"w(\d+)a(\d+):(\d+)", token)
        if not m:
            raise ValueError(f"bad schedule token '{token}', expected like 'w4a4:6'")
        wb, ab, epochs = (int(g) for g in m.groups())
        stages.append({"name": f"w{wb}a{ab}", "weight_bit_width": wb, "act_bit_width": ab,
                       "epochs": epochs, "lr": lr, "min_accuracy": min_acc})
    return stages


_RAW = argparse.RawDescriptionHelpFormatter


_BUILD_PY_TEMPLATE = '''"""FINN dataflow build for an axionlab-exported model.

Run inside the FINN Docker image (from this folder):
    ./run-docker.sh python build.py

A custom build step dumps each layer's MultiThreshold thresholds to
finn_build/thresholds/ DURING the build (right after streamlining, before the
thresholds are folded into hardware layers). These are the FloPoCo encoder input.
"""
import csv
import os

import numpy as np
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as cfg

HERE = os.path.dirname(os.path.abspath(__file__))


def step_dump_thresholds(model, build_cfg):
    from copy import deepcopy
    # round/clip to the integer thresholds the hardware actually uses (matches
    # FINN's >= comparison on integer accumulators) -> ready for a FloPoCo encoder.
    try:
        from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
        dump = deepcopy(model).transform(RoundAndClipThresholds())
    except Exception:
        dump = model
    out = os.path.join(build_cfg.output_dir, "thresholds")
    os.makedirs(out, exist_ok=True)

    def to_int(a):
        # smallest power-of-two scale that makes every threshold integer
        for F in range(0, 24):
            s = 2.0 ** F
            if np.allclose(a * s, np.round(a * s), atol=1e-4):
                return np.round(a * s).astype(np.int64), int(s)
        return np.round(a).astype(np.int64), 1

    def sbits(m):
        w = 2
        while (1 << (w - 1)) <= m:
            w += 1
        return w

    rows = []
    allf = open(os.path.join(out, "thresholds_all.txt"), "w")
    for i, node in enumerate(dump.graph.node):
        if node.op_type != "MultiThreshold":
            continue
        name = node.name or ("MultiThreshold_%d" % i)
        T = dump.get_initializer(node.input[1])
        idt = dump.get_tensor_datatype(node.input[0])    # encoder input dtype
        odt = dump.get_tensor_datatype(node.output[0])   # activation code dtype
        T2 = T if T.ndim == 2 else T.reshape(1, -1)
        Tint, scale = to_int(T2)                          # integer thresholds (+ input scale)
        ch, steps_ = Tint.shape
        if scale == 1 and idt is not None:
            win = idt.bitwidth()                         # integer accumulator -> use its width
        else:
            win = sbits(int(np.abs(Tint).max()) if Tint.size else 1)
        hdr = "%s | %d neurons | wIn=%d | wOut=%d thresholds | input_scale=%d (feed round(x*scale)) | out %s" % (
            name, ch, win, steps_, scale, odt.name)
        base = "%s_wIn%d_wOut%d_scale%d" % (name, win, steps_, scale)
        np.savetxt(os.path.join(out, base + ".txt"), Tint, fmt="%d")   # whole layer (rows = neurons)
        allf.write("# " + hdr + "\\n")
        np.savetxt(allf, Tint, fmt="%d")
        allf.write("\\n")

        # one file PER NEURON, integer thresholds as a column -> FloPoCo encoder input
        ndir = os.path.join(out, base)
        os.makedirs(ndir, exist_ok=True)
        for c in range(ch):
            np.savetxt(os.path.join(ndir, "neuron_%03d.txt" % c), Tint[c], fmt="%d")

        rows.append((name, ch, win, steps_, scale, idt.name if idt is not None else "?", odt.name))
    allf.close()
    with open(os.path.join(out, "thresholds_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "neurons", "wIn", "wOut", "input_scale", "in_dtype", "out_dtype"])
        w.writerows(rows)
    print("[step_dump_thresholds] wrote %d MultiThreshold layers -> %s/" % (len(rows), out))

    # also dump the integer WEIGHTS of each MatMul/Conv layer (w4 etc.)
    wdir = os.path.join(build_cfg.output_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    wrows = []
    for i, node in enumerate(dump.graph.node):
        if node.op_type not in ("MatMul", "Conv"):
            continue
        if len(node.input) < 2:
            continue
        W = dump.get_initializer(node.input[1])
        if W is None:
            continue
        wdt = dump.get_tensor_datatype(node.input[1])
        name = node.name or ("%s_%d" % (node.op_type, i))
        W2 = W if W.ndim == 2 else W.reshape(W.shape[0], -1)
        np.savetxt(os.path.join(wdir, name + ".txt"), W2, fmt="%.6g",
                   header="%s | shape %s | dtype %s" % (name, tuple(W.shape), wdt.name))
        wrows.append((name, str(tuple(W.shape)), wdt.name))
    with open(os.path.join(wdir, "weights_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "shape", "dtype"])
        w.writerows(wrows)
    print("[step_dump_thresholds] wrote %d weight tensors -> %s/" % (len(wrows), wdir))
    return model  # return the unmodified model; rounding was only for the dump


def step_dump_thresholds_final(model, build_cfg):
    # AFTER minimize_bit_width: read the REAL (clipped, narrow-int) thresholds from
    # the MVAU layers -> these match the hardware exactly. wIn = threshold datatype
    # width (e.g. 14/12), not the pre-minimize INT32 of step_dump_thresholds.
    out = os.path.join(build_cfg.output_dir, "thresholds_final")
    os.makedirs(out, exist_ok=True)
    rows = []
    for i, node in enumerate(model.graph.node):
        if not any(k in node.op_type for k in ("MVAU", "MatrixVectorActivation", "VVAU", "Thresholding")):
            continue
        # among the node's initializers, weights are 4-bit and thresholds are wider;
        # pick the widest-datatype initializer as the thresholds tensor.
        best = None
        for inp in node.input:
            init = model.get_initializer(inp)
            if init is None:
                continue
            dt = model.get_tensor_datatype(inp)
            bw = dt.bitwidth() if dt is not None else 0
            if best is None or bw > best[2]:
                best = (init, dt, bw)
        if best is None or best[2] <= 4:      # only weights -> matmul-only layer, no thresholds
            continue
        init, dt, bw = best
        T2 = init.reshape(-1, init.shape[-1]) if init.ndim >= 2 else init.reshape(1, -1)
        ch, steps_ = T2.shape
        name = node.name or ("Layer_%d" % i)
        base = "%s_wIn%d_wOut%d" % (name, bw, steps_)
        np.savetxt(os.path.join(out, base + ".txt"), T2, fmt="%d")
        ndir = os.path.join(out, base)
        os.makedirs(ndir, exist_ok=True)
        for c in range(ch):
            np.savetxt(os.path.join(ndir, "neuron_%03d.txt" % c), T2[c], fmt="%d")
        rows.append((name, ch, bw, steps_, dt.name if dt is not None else "?"))
    with open(os.path.join(out, "thresholds_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "neurons", "wIn", "wOut", "thr_dtype"])
        w.writerows(rows)
    print("[step_dump_thresholds_final] wrote %d MVAU threshold sets -> %s/" % (len(rows), out))
    return model


# dump pre-minimize thresholds after streamlining, and the REAL (clipped) ones after minimize
try:
    steps = list({steps_base})
except AttributeError:
    steps = list(cfg.default_build_dataflow_steps)
if "step_streamline" in steps:
    steps.insert(steps.index("step_streamline") + 1, step_dump_thresholds)
else:
    steps.append(step_dump_thresholds)
if "step_minimize_bit_width" in steps:
    steps.insert(steps.index("step_minimize_bit_width") + 1, step_dump_thresholds_final)
else:
    steps.append(step_dump_thresholds_final)

build.build_dataflow_cfg(
    os.path.join(HERE, "model.qonnx.onnx"),
    cfg.DataflowBuildConfig(
        output_dir=os.path.join(HERE, "finn_build"),
        synth_clk_period_ns=10.0,
        {target}
        {shell}steps=steps,
        generate_outputs={outputs},
    ),
)
print("done -> finn_build/ (synth level: {synth})")
'''

# synth level -> (base step list expr, generate_outputs expr)
_SYNTH_LEVELS = {
    "estimate": ("cfg.estimate_only_dataflow_steps",
                 "[cfg.DataflowOutputType.ESTIMATE_REPORTS]"),
    "ip": ("cfg.default_build_dataflow_steps",
           "[cfg.DataflowOutputType.ESTIMATE_REPORTS, cfg.DataflowOutputType.OOC_SYNTH, "
           "cfg.DataflowOutputType.STITCHED_IP, cfg.DataflowOutputType.RTLSIM_PERFORMANCE]"),
    "bitfile": ("cfg.default_build_dataflow_steps",
                "[cfg.DataflowOutputType.ESTIMATE_REPORTS, cfg.DataflowOutputType.STITCHED_IP, "
                "cfg.DataflowOutputType.OOC_SYNTH, cfg.DataflowOutputType.BITFILE, "
                "cfg.DataflowOutputType.PYNQ_DRIVER, cfg.DataflowOutputType.DEPLOYMENT_PACKAGE]"),
}


def parse_shape(spec: str) -> tuple:
    """'3,224,224' -> (3, 224, 224); '16' -> (16,)."""
    return tuple(int(x) for x in spec.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", formatter_class=_RAW,
        description="axionlab — build / train / quantize / deploy quantized neural nets.",
        epilog=(
            "commands:\n"
            "  train   train a model (optionally quantized) from an experiment yaml\n"
            "  qat     progressive quantization-aware training with rollback\n"
            "  infer   run inference (image / webcam / dataset), optional GUI\n"
            "  eval    evaluate a checkpoint on the val/test set\n"
            "  export  export a (quantized) model to QONNX for FINN\n"
            "  finn    build an FPGA dataflow accelerator from a QONNX file\n\n"
            "see per-command help, e.g.:  python run.py train -h\n"
            "full docs: docs/README.md"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    def common(sp):
        sp.add_argument("--exp", "--experiment", dest="exp", required=True,
                        metavar="YAML", help="experiment yaml (defines model, dataset, training)")
        sp.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                        help="compute device (default: auto-detect CUDA/ROCm > MPS > CPU)")

    # train
    t = sub.add_parser(
        "train", help="train a model", formatter_class=_RAW,
        description="Train a model defined by an experiment yaml. CLI flags override the yaml.",
        epilog=(
            "examples:\n"
            "  python run.py train --exp configs/experiments/jsc/jsc_2l_qat.yaml \\\n"
            "      --epochs 10 --lr 0.03 --device cuda --dashboard \\\n"
            "      --early-stop plateau --patience 5 --dry-run\n\n"
            "  # quick pipeline smoke test (1 epoch, few batches, no download wait):\n"
            "  python run.py train --exp configs/experiments/jsc/jsc_2l_qat.yaml --smoke"
        ),
    )
    common(t)
    t.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    t.add_argument("--lr", type=float, default=None, help="override learning rate")
    t.add_argument("--track-db", dest="track_db", default=None, metavar="PATH",
                   help="record metrics+weights+yamls into this SQLite DB")
    t.add_argument("--early-stop", dest="early_stop", choices=["plateau", "none"], default=None,
                   help="stop when the best metric plateaus")
    t.add_argument("--patience", type=int, default=None, help="epochs without improvement before early-stop")
    t.add_argument("--dashboard", action="store_true", help="show a live Rich dashboard")
    t.add_argument("--dry-run", dest="dry_run", action="store_true", help="train but write no checkpoints")
    t.add_argument("--smoke", action="store_true", help="tiny fast run (1 epoch, few batches)")
    t.set_defaults(func=cmd_train)

    # qat
    q = sub.add_parser(
        "qat", help="progressive QAT with rollback", formatter_class=_RAW,
        description=("Progressive quantization-aware training: optional float warmup, then lower the "
                     "bit width in stages. A stage that misses --min-acc rolls back to the previous "
                     "bit width and stops. Weights and activations are scheduled independently."),
        epilog=(
            "schedule format: comma-separated  wWaA:EPOCHS  (W=weight bits, A=act bits)\n\n"
            "examples:\n"
            "  python run.py qat --exp configs/experiments/jsc/jsc_2l_qat.yaml \\\n"
            "      --warmup-epochs 5 --schedule w8a8:6,w4a4:6,w2a2:8 --min-acc 0.7 --dashboard\n\n"
            "  # lower weights first, then activations:\n"
            "  python run.py qat --exp E.yaml --schedule w8a8:6,w4a8:6,w4a4:6"
        ),
    )
    common(q)
    q.add_argument("--warmup-epochs", dest="warmup_epochs", type=int, default=0,
                   help="float epochs before quantizing (0 = none)")
    q.add_argument("--schedule", required=True, metavar="wWaA:E,...",
                   help="bit-width stages, e.g. w8a8:6,w4a4:6,w2a2:8")
    q.add_argument("--lr", type=float, default=1e-4, help="per-stage learning rate")
    q.add_argument("--min-acc", dest="min_acc", type=float, default=None, metavar="X",
                   help="roll back + stop if a stage's best metric is below X")
    q.add_argument("--track-db", dest="track_db", default=None, metavar="PATH",
                   help="record into this SQLite DB")
    q.add_argument("--dashboard", action="store_true", help="show a live Rich dashboard")
    q.set_defaults(func=cmd_qat)

    # infer
    i = sub.add_parser(
        "infer", help="run inference", formatter_class=_RAW,
        description="Load a checkpoint and run inference on an image, the webcam, or a dataset.",
        epilog=(
            "examples:\n"
            "  python run.py infer --exp configs/models/resnet18.yaml --weights best.pth \\\n"
            "      --source webcam --gui\n"
            "  python run.py infer --exp configs/models/resnet18.yaml --weights best.pth \\\n"
            "      --source path/to/image.jpg"
        ),
    )
    common(i)
    i.add_argument("--weights", default="", metavar="PTH", help="checkpoint .pth to load")
    i.add_argument("--source", default="dataset", metavar="SRC",
                   help="'webcam' | an image path | 'dataset'")
    i.add_argument("--gui", action="store_true", help="open the interactive inference GUI")
    i.add_argument("--save-dir", dest="save_dir", default="screenshots",
                   help="where the GUI saves screenshots")
    i.set_defaults(func=cmd_infer)

    # eval
    e = sub.add_parser(
        "eval", help="evaluate a checkpoint", formatter_class=_RAW,
        description="Load a checkpoint and report loss/accuracy on the val/test split.",
        epilog="example:\n  python run.py eval --exp E.yaml --weights best.pth",
    )
    common(e)
    e.add_argument("--weights", default="best.pth", metavar="PTH", help="checkpoint to evaluate")
    e.set_defaults(func=cmd_eval)

    # export
    x = sub.add_parser(
        "export", help="export a (quantized) model to QONNX", formatter_class=_RAW,
        description="Build the net from its yaml, load weights, optionally quantize the input, "
                    "and export QONNX for the FINN flow.",
        epilog=("example:\n"
                "  python run.py export --model configs/models/jsc_2l.yaml --weights best.pth \\\n"
                "      --input-shape 16 --input-quant 8 --out jsc.qonnx.onnx"),
    )
    x.add_argument("--model", required=True, metavar="YAML", help="net yaml (the architecture)")
    x.add_argument("--weights", default="", metavar="PTH", help="checkpoint to load")
    x.add_argument("--input-shape", dest="input_shape", required=True, metavar="C,H,W",
                   help="input shape without batch, e.g. 3,224,224 or 16")
    x.add_argument("--input-quant", dest="input_quant", type=int, default=None, metavar="BITS",
                   help="wrap the input in a QuantIdentity of this bit width")
    x.add_argument("--quantize", default=None, metavar="wWaA",
                   help="rebuild the net as quant (e.g. w4a4) before loading weights "
                        "(needed when the checkpoint came from quantize-and-finetune surgery)")
    x.add_argument("--out", default="model.qonnx.onnx", help="output .onnx path")
    x.add_argument("--bundle", default=None, metavar="DIR",
                   help="also assemble a self-contained FINN build folder (qonnx + build.py "
                        "+ model yaml + weights + provenance) you can run in the FINN Docker")
    x.add_argument("--fpga-part", dest="fpga_part", default="xc7z020clg400-1",
                   help="FPGA part baked into the bundle's build.py")
    x.add_argument("--board", default=None,
                   help="target board name (e.g. Pynq-Z2); overrides --fpga-part for a bitfile")
    x.add_argument("--synth", choices=["estimate", "ip", "bitfile"], default="estimate",
                   help="how far the bundle's build.py goes: estimate (no Vivado) | "
                        "ip (HLS+synth+stitched IP, needs Vivado) | bitfile (full bitstream)")
    x.set_defaults(func=cmd_export)

    # finn
    f = sub.add_parser(
        "finn", help="build an FPGA accelerator from QONNX", formatter_class=_RAW,
        description="Run the FINN dataflow build on a QONNX model (inside the FINN Docker image).",
        epilog=("example:\n"
                "  python run.py finn jsc.qonnx.onnx --fpga-part xc7z020clg400-1 --output build"),
    )
    f.add_argument("qonnx", help="QONNX model path (from `export`)")
    f.add_argument("--output", default="finn_build", help="FINN build output dir")
    f.add_argument("--fpga-part", dest="fpga_part", default="xc7z020clg400-1")
    f.add_argument("--clk-ns", dest="clk_ns", type=float, default=10.0, help="target clock period (ns)")
    f.add_argument("--no-rtlsim", dest="no_rtlsim", action="store_true",
                   help="skip stitched-IP / rtlsim outputs (estimates only)")
    f.set_defaults(func=cmd_finn)

    return p


# ---- handlers (lazy heavy imports) ----------------------------------------

def _overrides(args) -> dict:
    ov = {"epochs": getattr(args, "epochs", None), "lr": getattr(args, "lr", None),
          "device": getattr(args, "device", None), "tracking_db": getattr(args, "track_db", None)}
    if getattr(args, "dry_run", False):
        ov["save_best"] = False
        ov["save_last"] = False
    if getattr(args, "smoke", False):
        ov["epochs"] = 1
    return ov


def _dashboard(args):
    if getattr(args, "dashboard", False):
        from src.ui import RichDashboardCallback
        return [RichDashboardCallback(args.exp)]
    return []


def cmd_train(args):
    from src.core import Runner

    runner = Runner.from_experiment(args.exp, overrides=_overrides(args))
    if args.smoke:
        runner.max_steps = 5
    if args.early_stop == "plateau":
        runner.config.early_stopping_patience = args.patience or 5
    runner.train(callbacks=_dashboard(args))
    print("eval:", runner.evaluate())


def cmd_qat(args):
    from src.core import Phase, Runner
    from src.core.qat_schedule import QatStage, run_qat_schedule

    runner = Runner.from_experiment(args.exp, overrides=_overrides(args))
    cbs = _dashboard(args)
    if args.warmup_epochs:
        runner.train(plan=[Phase("float_warmup", epochs=args.warmup_epochs, lr=args.lr)], callbacks=cbs)
    stages = [QatStage(**s) for s in parse_schedule(args.schedule, args.lr, args.min_acc)]
    result = run_qat_schedule(runner, stages, callbacks=cbs)
    print("schedule result:", result)
    print("eval:", runner.evaluate())


def cmd_infer(args):
    from src.config.reader import ConfigReader
    from src.inference import load_classifier

    cfg = ConfigReader.read_config(args.exp, validate=False)
    # accept either an experiment yaml (has `model:`) or a model/net yaml directly
    model_yaml = cfg["model"] if "model" in cfg else args.exp
    predictor = load_classifier(model_yaml, args.weights, device=args.device or "cpu")

    if args.gui:
        from src.gui import InferenceGUI
        InferenceGUI(predictor).run()
        return

    if args.source == "webcam":
        from src.gui import InferenceGUI
        gui = InferenceGUI(predictor)
        gui.toggle_webcam()
        gui.run()
    else:
        from PIL import Image
        pred = predictor.predict(Image.open(args.source).convert("RGB"))
        print(f"{pred.label}  ({pred.score:.3f})")


def cmd_eval(args):
    from src.core import Runner

    runner = Runner.from_experiment(args.exp, overrides=_overrides(args))
    print("eval:", runner.evaluate(args.weights))


def cmd_export(args):
    import re

    import torch

    from src.core import build_network
    from src.export import add_input_quant, export_to_qonnx

    shape = parse_shape(args.input_shape)
    model = build_network(args.model)
    if args.quantize:  # reconstruct the quant architecture so the checkpoint loads
        m = re.fullmatch(r"w(\d+)a(\d+)", args.quantize)
        if not m:
            raise ValueError(f"--quantize expects 'wWaA' (e.g. w4a4), got '{args.quantize}'")
        from src.core import set_bit_width
        set_bit_width(model, int(m.group(1)), int(m.group(2)))
    if args.weights:
        # brevitas creates activation-scale params lazily on first forward; run a
        # dummy pass so the trained checkpoint's keys match on a strict load.
        model.train()
        with torch.no_grad():
            model(torch.zeros(1, *shape))
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    if args.input_quant:
        model = add_input_quant(model, bit_width=args.input_quant)
    import os
    out_dir = os.path.dirname(args.out)          # --out may point into a not-yet-existing folder
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    export_to_qonnx(model, shape, args.out)
    print("wrote", args.out)

    if args.bundle:
        import json
        import os
        import shutil

        os.makedirs(args.bundle, exist_ok=True)
        bundle_onnx = os.path.join(args.bundle, "model.qonnx.onnx")
        # --out may already be that exact file (e.g. --out <bundle>/model.qonnx.onnx)
        if os.path.abspath(args.out) != os.path.abspath(bundle_onnx):
            shutil.copy(args.out, bundle_onnx)
        shutil.copy(args.model, os.path.join(args.bundle, os.path.basename(args.model)))
        if args.weights and os.path.exists(args.weights):
            shutil.copy(args.weights, os.path.join(args.bundle, os.path.basename(args.weights)))
        steps_base, outputs = _SYNTH_LEVELS[args.synth]
        if args.synth == "bitfile":
            # a full Zynq bitstream needs a board (sets the part) + a shell flow type
            board = args.board or "Pynq-Z2"
            target = f'board="{board}",'
            shell = "shell_flow_type=cfg.ShellFlowType.VIVADO_ZYNQ,\n        "
        else:
            target = (f'board="{args.board}",' if args.board else f'fpga_part="{args.fpga_part}",')
            shell = ""
        with open(os.path.join(args.bundle, "build.py"), "w") as f:
            f.write(_BUILD_PY_TEMPLATE.format(steps_base=steps_base, outputs=outputs,
                                              target=target, shell=shell, synth=args.synth))
        with open(os.path.join(args.bundle, "info.json"), "w") as f:
            json.dump({"model_yaml": args.model, "weights": args.weights,
                       "quantize": args.quantize, "input_shape": args.input_shape,
                       "input_quant": args.input_quant, "fpga_part": args.fpga_part,
                       "board": args.board, "synth": args.synth}, f, indent=2)
        # drop a run-docker.sh so the bundle is self-contained: ./run-docker.sh python build.py
        from src.webapp.finn_docker import ensure_run_docker
        ensure_run_docker(args.bundle)
        print(f"bundle ready: {args.bundle}/  (model.qonnx.onnx, build.py, run-docker.sh, info.json, configs, weights)")


def cmd_finn(args):
    try:
        from src.finn import build_finn
        out = build_finn(args.qonnx, output_dir=args.output, fpga_part=args.fpga_part,
                         clk_ns=args.clk_ns, rtlsim=not args.no_rtlsim)
        print("FINN build in", out)
    except ModuleNotFoundError as e:
        if "finn" in str(e):
            print("FINN is not installed in this environment. The FINN build only runs "
                  "inside the FINN Docker image — run this command there (the QONNX file "
                  "from `export` is the input).")
        else:
            raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
