"""FINN dataflow build for an axionlab-exported model.

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
        allf.write("# " + hdr + "\n")
        np.savetxt(allf, Tint, fmt="%d")
        allf.write("\n")

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
    steps = list(cfg.default_build_dataflow_steps)
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
        fpga_part="xc7z020clg400-1",
        steps=steps,
        generate_outputs=[cfg.DataflowOutputType.ESTIMATE_REPORTS, cfg.DataflowOutputType.OOC_SYNTH, cfg.DataflowOutputType.STITCHED_IP, cfg.DataflowOutputType.RTLSIM_PERFORMANCE],
    ),
)
print("done -> finn_build/ (synth level: ip)")
