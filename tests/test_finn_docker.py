"""Spec for the FINN docker orchestration. Command building is pure; preflight is
structural (we don't require docker/Vivado to be installed)."""

import os

from src.webapp import finn_docker


def test_docker_command_basics(tmp_path):
    argv = finn_docker.docker_command(str(tmp_path), image="finn:test",
                                      command="python build.py")
    assert argv[0] == "docker" and "run" in argv and "--rm" in argv
    # bundle is mounted at /work and used as workdir
    joined = " ".join(argv)
    assert f"{tmp_path.resolve()}:/work" in joined
    assert "-w" in argv and "/work" in argv
    assert argv[-3:] == ["finn:test", "python", "build.py"]


def test_docker_command_mounts_xilinx(tmp_path, monkeypatch):
    monkeypatch.setenv("FINN_XILINX_PATH", "/opt/Xilinx")
    monkeypatch.setenv("FINN_XILINX_VERSION", "2022.1")
    argv = finn_docker.docker_command(str(tmp_path), image="finn:test")
    joined = " ".join(argv)
    assert "/opt/Xilinx:/opt/Xilinx:ro" in joined
    assert "FINN_XILINX_PATH=/opt/Xilinx" in joined
    assert "FINN_XILINX_VERSION=2022.1" in joined


def test_ensure_run_docker_writes_executable(tmp_path):
    path = finn_docker.ensure_run_docker(str(tmp_path), image="finn:test")
    p = tmp_path / "run-docker.sh"
    assert p.exists()
    assert os.access(p, os.X_OK)
    text = p.read_text()
    assert "docker run" in text and "finn:test" in text
    # idempotent: doesn't clobber an existing one
    p.write_text("# custom\n")
    finn_docker.ensure_run_docker(str(tmp_path))
    assert p.read_text() == "# custom\n"


def test_preflight_structure():
    pf = finn_docker.preflight(image="finn:test", synth="estimate")
    names = {c["name"] for c in pf["checks"]}
    assert names == {"docker", "image", "xilinx_tools"}
    assert "ok" in pf and "blocking" in pf
    # estimate doesn't require Vivado -> the xilinx check is non-blocking
    xil = next(c for c in pf["checks"] if c["name"] == "xilinx_tools")
    assert xil["ok"] is True


def test_preflight_bitfile_requires_vivado(monkeypatch):
    monkeypatch.delenv("FINN_XILINX_PATH", raising=False)
    pf = finn_docker.preflight(image="finn:test", synth="bitfile")
    xil = next(c for c in pf["checks"] if c["name"] == "xilinx_tools")
    assert xil["ok"] is False and "xilinx_tools" in pf["blocking"]
