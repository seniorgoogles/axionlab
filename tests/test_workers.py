"""Spec for distributed-job workers: registry, routing, capabilities. Pure sqlite."""

import pytest

from src.webapp import workers
from src.webapp.workers import WorkerRegistry


def test_register_and_list(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    reg.register("gpu-box", "http://10.0.0.5:8000/", ["docker", "gpu"])
    reg.register("fpga-box", "http://10.0.0.6:8000", ["docker", "vivado"])

    names = {w["name"] for w in reg.list()}
    assert names == {"gpu-box", "fpga-box"}
    gpu = reg.get("gpu-box")
    assert gpu["url"] == "http://10.0.0.5:8000"          # trailing slash stripped
    assert gpu["capabilities"] == ["docker", "gpu"]


def test_register_replaces(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    reg.register("box", "http://a:8000", ["docker"])
    reg.register("box", "http://b:8000", ["docker", "vivado"])
    assert len(reg.list()) == 1
    assert reg.get("box")["url"] == "http://b:8000"


def test_remove(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    reg.register("box", "http://a:8000")
    assert reg.remove("box") is True
    assert reg.remove("box") is False


def test_requires_name_and_url(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    with pytest.raises(ValueError):
        reg.register("", "http://a:8000")
    with pytest.raises(ValueError):
        reg.register("box", "")


def test_route_finn_goes_to_vivado(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    reg.register("gpu-box", "http://a:8000", ["docker", "gpu"])
    reg.register("fpga-box", "http://b:8000", ["docker", "vivado"])
    assert reg.route("finn")["name"] == "fpga-box"      # finn needs vivado


def test_route_unknown_kind_picks_first(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    reg.register("a-box", "http://a:8000", ["gpu"])
    reg.register("b-box", "http://b:8000", ["vivado"])
    # 'train' has no capability requirement -> first worker
    assert reg.route("train")["name"] == "a-box"


def test_route_no_capable_worker(tmp_path):
    reg = WorkerRegistry(str(tmp_path / "w.db"))
    reg.register("gpu-box", "http://a:8000", ["gpu"])    # no vivado anywhere
    assert reg.route("finn") is None


def test_pick_worker_helper():
    ws = [{"name": "a", "capabilities": ["gpu"]}, {"name": "b", "capabilities": ["vivado"]}]
    assert workers.pick_worker(ws, "vivado")["name"] == "b"
    assert workers.pick_worker(ws, None)["name"] == "a"
    assert workers.pick_worker([], "gpu") is None


def test_detect_capabilities_is_a_list():
    caps = workers.detect_capabilities()
    assert isinstance(caps, list)
    assert all(c in ("docker", "gpu", "vivado") for c in caps)
