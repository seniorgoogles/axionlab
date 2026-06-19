"""Run the FINN dataflow build on an exported QONNX model.

This closes the loop: axionlab/Brevitas QAT -> QONNX (src.export) -> FINN -> FPGA.
FINN only runs inside its Docker image (needs Vivado/Vitis), so this is a thin,
well-documented wrapper around finn.builder.build_dataflow.

    # inside the FINN docker:
    from src.finn import build_finn
    build_finn("model.qonnx.onnx", output_dir="build", fpga_part="xc7z020clg400-1")

Or as a script:
    python -m src.finn.build_finn model.qonnx.onnx --output build --fpga-part xc7z020clg400-1
"""

from __future__ import annotations


def build_finn(qonnx_path: str, output_dir: str = "finn_build",
               fpga_part: str = "xc7z020clg400-1", clk_ns: float = 10.0,
               rtlsim: bool = True):
    """Compile a QONNX model into a FINN dataflow accelerator.

    Args:
        qonnx_path: path to the QONNX model (from src.export.export_to_qonnx).
        output_dir: FINN build output directory.
        fpga_part: target FPGA part.
        clk_ns: target clock period in ns.
        rtlsim: also produce + verify via rtlsim (stitched-IP) outputs.
    """
    # imports are local: only available inside the FINN docker
    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as cfg

    outputs = [cfg.DataflowOutputType.ESTIMATE_REPORTS]
    if rtlsim:
        outputs += [cfg.DataflowOutputType.STITCHED_IP, cfg.DataflowOutputType.RTLSIM_PERFORMANCE]

    build_cfg = cfg.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=clk_ns,
        fpga_part=fpga_part,
        generate_outputs=outputs,
    )
    build.build_dataflow_cfg(qonnx_path, build_cfg)
    return output_dir


def _main():
    import argparse

    p = argparse.ArgumentParser(description="Run FINN dataflow build on a QONNX model.")
    p.add_argument("qonnx_path")
    p.add_argument("--output", default="finn_build")
    p.add_argument("--fpga-part", default="xc7z020clg400-1")
    p.add_argument("--clk-ns", type=float, default=10.0)
    p.add_argument("--no-rtlsim", action="store_true")
    args = p.parse_args()
    build_finn(args.qonnx_path, args.output, args.fpga_part, args.clk_ns, not args.no_rtlsim)


if __name__ == "__main__":
    _main()
