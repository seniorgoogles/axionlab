"""Spec for the Projects backbone. Pure sqlite -- no torch/fastapi needed."""

import pytest

from src.webapp.projects import ARTIFACT_KINDS, ProjectDB


def test_project_roundtrip(tmp_path):
    db = ProjectDB(str(tmp_path / "p.db"))
    pid = db.create_project("jsc-2l", description="2-layer JSC")

    proj = db.get_project(pid)
    assert proj["name"] == "jsc-2l" and proj["description"] == "2-layer JSC"
    # every artifact kind is present as an (empty) group
    assert set(proj["artifacts"]) == set(ARTIFACT_KINDS)
    assert all(proj["artifacts"][k] == [] for k in ARTIFACT_KINDS)

    listed = db.list_projects()
    assert len(listed) == 1 and listed[0]["artifact_count"] == 0


def test_artifacts_grouped_by_kind(tmp_path):
    db = ProjectDB(str(tmp_path / "p.db"))
    pid = db.create_project("proj")
    base = db.add_artifact(pid, "model", "resnet18.yaml", path="configs/models/resnet18.yaml")
    db.add_artifact(pid, "quant_model", "best.pth", meta={"quantize": "w4a4"}, parent_id=base)
    db.add_artifact(pid, "training_data", "cifar10")
    db.add_artifact(pid, "hardware_model", "Pynq-Z2", meta={"fpga_part": "xc7z020clg400-1"})

    proj = db.get_project(pid)
    assert len(proj["artifacts"]["model"]) == 1
    q = proj["artifacts"]["quant_model"][0]
    assert q["meta"] == {"quantize": "w4a4"} and q["parent_id"] == base
    assert proj["artifacts"]["hardware_model"][0]["meta"]["fpga_part"] == "xc7z020clg400-1"
    assert db.list_projects()[0]["artifact_count"] == 4


def test_unknown_kind_rejected(tmp_path):
    db = ProjectDB(str(tmp_path / "p.db"))
    pid = db.create_project("proj")
    with pytest.raises(ValueError):
        db.add_artifact(pid, "not_a_kind", "x")


def test_artifact_on_missing_project_rejected(tmp_path):
    db = ProjectDB(str(tmp_path / "p.db"))
    with pytest.raises(ValueError):
        db.add_artifact(999, "model", "x")


def test_delete_project_cascades_artifacts(tmp_path):
    db = ProjectDB(str(tmp_path / "p.db"))
    pid = db.create_project("proj")
    db.add_artifact(pid, "model", "m")
    assert db.delete_project(pid) is True
    assert db.get_project(pid) is None
    # cascade removed the artifact too
    assert db.conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"] == 0


def test_delete_artifact(tmp_path):
    db = ProjectDB(str(tmp_path / "p.db"))
    pid = db.create_project("proj")
    aid = db.add_artifact(pid, "model", "m")
    assert db.delete_artifact(aid) is True
    assert db.delete_artifact(aid) is False
    assert db.get_project(pid)["artifacts"]["model"] == []
