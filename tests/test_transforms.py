"""Spec for transformation tracking (lineage). Pure sqlite -- no torch needed."""

import pytest

from src.webapp.transforms import TransformDB


def test_record_and_get_run(tmp_path):
    db = TransformDB(str(tmp_path / "t.db"))
    db.record_step("run1", 0, "qat", input="float.pth", output="quant.pth",
                   params={"schedule": "w8a8:6,w4a4:6"}, project_id=1)
    db.record_step("run1", 1, "export", input="quant.pth", output="model.qonnx.onnx",
                   snapshot="snap/0_export.onnx", project_id=1)

    steps = db.get_run("run1")
    assert [s["step"] for s in steps] == ["qat", "export"]
    assert steps[0]["params"]["schedule"] == "w8a8:6,w4a4:6"
    assert steps[1]["snapshot"] == "snap/0_export.onnx"


def test_start_then_finish_step(tmp_path):
    db = TransformDB(str(tmp_path / "t.db"))
    sid = db.start_step("run1", 0, "finn:streamline", input="model.qonnx.onnx")
    assert db.get_run("run1")[0]["status"] == "running"
    db.finish_step(sid, status="ok", output="streamlined.onnx", snapshot="snap/0.onnx")
    step = db.get_run("run1")[0]
    assert step["status"] == "ok" and step["output"] == "streamlined.onnx"
    assert step["ended"] is not None


def test_first_failed_pins_the_step(tmp_path):
    db = TransformDB(str(tmp_path / "t.db"))
    db.record_step("run1", 0, "qat", status="ok")
    db.record_step("run1", 1, "export", status="ok")
    db.record_step("run1", 2, "finn:minimize_bit_width", status="failed",
                   error="bit width mismatch")
    db.record_step("run1", 3, "finn:synth", status="ok")  # later, ignored

    failed = db.first_failed("run1")
    assert failed["seq"] == 2 and failed["step"] == "finn:minimize_bit_width"
    assert "bit width" in failed["error"]


def test_run_summary_status_rolls_up(tmp_path):
    db = TransformDB(str(tmp_path / "t.db"))
    db.record_step("ok_run", 0, "qat", status="ok", project_id=1)
    db.record_step("bad_run", 0, "qat", status="ok", project_id=1)
    db.record_step("bad_run", 1, "export", status="failed", project_id=1)

    runs = {r["run_id"]: r for r in db.list_runs(project_id=1)}
    assert runs["ok_run"]["status"] == "ok" and runs["ok_run"]["steps"] == 1
    assert runs["bad_run"]["status"] == "failed" and runs["bad_run"]["failed"] == 1


def test_bad_status_rejected(tmp_path):
    db = TransformDB(str(tmp_path / "t.db"))
    with pytest.raises(ValueError):
        db.record_step("run1", 0, "qat", status="not_a_status")
