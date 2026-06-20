"""Spec for the ONNX viewer/diff. Builds tiny ONNX graphs with the onnx package
directly (no torch). Skips cleanly if onnx isn't installed."""

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from src.webapp import onnxview  # noqa: E402


def _make(path, op_types):
    """A linear graph X -> op0 -> op1 -> ... -> Y, one node per op_type."""
    nodes, prev = [], "X"
    for i, op in enumerate(op_types):
        out = "Y" if i == len(op_types) - 1 else f"t{i}"
        nodes.append(helper.make_node(op, [prev], [out], name=f"{op}_{i}"))
        prev = out
    g = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])],
        initializer=[helper.make_tensor("w", TensorProto.INT8, [2, 2], [1, 2, 3, 4])],
    )
    m = helper.make_model(g)
    onnx.save(m, str(path))


def test_summary(tmp_path):
    _make(tmp_path / "m.onnx", ["Relu", "MatMul"])
    s = onnxview.summary(str(tmp_path), "m.onnx")
    assert s["n_nodes"] == 2
    assert s["op_counts"] == {"Relu": 1, "MatMul": 1}
    assert s["inputs"][0]["name"] == "X" and s["inputs"][0]["dtype"] == "FLOAT"
    assert s["n_initializers"] == 1
    assert s["initializers"][0]["dtype"] == "INT8" and s["initializers"][0]["shape"] == [2, 2]


def test_diff_added_removed_and_delta(tmp_path):
    _make(tmp_path / "a.onnx", ["Relu", "MatMul"])
    _make(tmp_path / "b.onnx", ["Relu", "MultiThreshold"])     # MatMul -> MultiThreshold
    d = onnxview.diff(str(tmp_path), "a.onnx", "b.onnx")
    assert d["op_delta"].get("MatMul") == -1
    assert d["op_delta"].get("MultiThreshold") == 1
    assert any(n["op_type"] == "MultiThreshold" for n in d["added"])
    assert any(n["op_type"] == "MatMul" for n in d["removed"])


def test_summary_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        onnxview.summary(str(tmp_path), "nope.onnx")


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        onnxview.summary(str(tmp_path), "../escape.onnx")
