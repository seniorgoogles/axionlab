"""Spec for the CLI (task #20): argument parsing + schedule parsing.

These are stdlib-only (no torch), so the parser/schedule logic is testable on its
own; the handlers (train/qat/infer) reuse Runner/run_qat_schedule/predictor.
"""

from src.cli import build_parser, parse_schedule, parse_shape


def test_train_args():
    args = build_parser().parse_args(
        ["train", "--exp", "e.yaml", "--epochs", "10", "--lr", "0.03",
         "--device", "cuda", "--dashboard", "--dry-run", "--early-stop", "plateau", "--patience", "5"])
    assert args.command == "train"
    assert args.exp == "e.yaml"
    assert args.epochs == 10
    assert args.lr == 0.03
    assert args.device == "cuda"
    assert args.dashboard is True
    assert args.dry_run is True
    assert args.early_stop == "plateau"
    assert args.patience == 5


def test_qat_args():
    args = build_parser().parse_args(
        ["qat", "--exp", "e.yaml", "--warmup-epochs", "5",
         "--schedule", "w8a8:6,w4a4:6,w2a2:8", "--min-acc", "0.7", "--dashboard"])
    assert args.command == "qat"
    assert args.warmup_epochs == 5
    assert args.schedule == "w8a8:6,w4a4:6,w2a2:8"
    assert args.min_acc == 0.7


def test_infer_args():
    args = build_parser().parse_args(
        ["infer", "--exp", "e.yaml", "--weights", "best.pth", "--source", "webcam", "--gui"])
    assert args.command == "infer"
    assert args.weights == "best.pth"
    assert args.source == "webcam"
    assert args.gui is True


def test_export_args():
    args = build_parser().parse_args(
        ["export", "--model", "configs/models/jsc_2l.yaml", "--weights", "best.pth",
         "--out", "m.qonnx.onnx", "--input-shape", "16", "--input-quant", "8"])
    assert args.command == "export"
    assert args.model == "configs/models/jsc_2l.yaml"
    assert args.input_shape == "16"
    assert args.input_quant == 8


def test_finn_args():
    args = build_parser().parse_args(
        ["finn", "m.qonnx.onnx", "--fpga-part", "xc7z020clg400-1", "--no-rtlsim"])
    assert args.command == "finn"
    assert args.qonnx == "m.qonnx.onnx"
    assert args.no_rtlsim is True


def test_parse_shape():
    assert parse_shape("3,224,224") == (3, 224, 224)
    assert parse_shape("16") == (16,)


def test_parse_schedule():
    stages = parse_schedule("w8a8:6,w4a4:4", lr=1e-3, min_acc=0.7)
    assert len(stages) == 2
    assert stages[0] == {"name": "w8a8", "weight_bit_width": 8, "act_bit_width": 8,
                         "epochs": 6, "lr": 1e-3, "min_accuracy": 0.7}
    assert stages[1]["weight_bit_width"] == 4 and stages[1]["epochs"] == 4
