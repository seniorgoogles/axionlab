"""Orchestrate the FINN dataflow build inside Docker.

FINN only runs inside its Docker image (it needs Vivado/Vitis), so the GUI can't
import it -- it has to launch a container. This module builds that `docker run`
command and preflight-checks the prerequisites, so the user gets a clear message
("docker missing", "image not pulled", "no Vivado path") instead of a cryptic
subprocess failure.

The container mounts the bundle dir (from `export --bundle`: model.qonnx.onnx +
build.py) at /work and runs `python build.py` there. If FINN_XILINX_PATH is set,
the Xilinx tools dir is mounted too so a real synthesis/bitfile build can find
Vivado; without it only the `estimate` level (no Vivado) makes sense.

Command construction is pure/testable; the actual run goes through JobManager so
the GUI streams its logs and can stop it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_IMAGE = os.environ.get("FINN_DOCKER_IMAGE", "maltanar/finn:latest")


# a self-contained run-docker.sh dropped into a bundle so `./run-docker.sh python
# build.py` works: it mounts the bundle (the script's own dir) at /work and the
# Xilinx tools if FINN_XILINX_PATH is set, then runs the given command in the image.
RUN_DOCKER_SH = """#!/usr/bin/env bash
# Run a command inside the FINN docker image, with this bundle mounted at /work.
#   ./run-docker.sh python build.py
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
IMAGE="${FINN_DOCKER_IMAGE:-%(image)s}"

ARGS=(docker run --rm -v "$HERE:/work" -w /work)
if [ -n "${FINN_XILINX_PATH:-}" ]; then
  ARGS+=(-v "$FINN_XILINX_PATH:$FINN_XILINX_PATH:ro" -e "FINN_XILINX_PATH=$FINN_XILINX_PATH")
  [ -n "${FINN_XILINX_VERSION:-}" ] && ARGS+=(-e "FINN_XILINX_VERSION=$FINN_XILINX_VERSION")
fi
ARGS+=("$IMAGE" "$@")
echo "+ ${ARGS[*]}"
exec "${ARGS[@]}"
"""


def ensure_run_docker(bundle_dir: str, image: str = DEFAULT_IMAGE) -> str:
    """Write run-docker.sh into the bundle if it's missing. Returns its path."""
    dest = Path(bundle_dir) / "run-docker.sh"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(RUN_DOCKER_SH % {"image": image})
        dest.chmod(0o755)
    return str(dest)


def docker_command(bundle_dir: str, image: str = DEFAULT_IMAGE,
                   command: str = "python build.py",
                   xilinx_path: Optional[str] = None,
                   gpus: bool = False) -> List[str]:
    """Build the `docker run` argv that runs `command` in the bundle at /work.

    xilinx_path: host dir of the Xilinx tools (Vivado/Vitis) to mount read-only;
                 defaults to $FINN_XILINX_PATH. Needed for ip/bitfile, not estimate.
    """
    bundle = str(Path(bundle_dir).resolve())
    argv = ["docker", "run", "--rm", "-v", f"{bundle}:/work", "-w", "/work"]
    if gpus:
        argv += ["--gpus", "all"]
    xpath = xilinx_path or os.environ.get("FINN_XILINX_PATH")
    if xpath:
        argv += ["-v", f"{xpath}:{xpath}:ro", "-e", f"FINN_XILINX_PATH={xpath}"]
        xver = os.environ.get("FINN_XILINX_VERSION")
        if xver:
            argv += ["-e", f"FINN_XILINX_VERSION={xver}"]
    argv += [image] + command.split()
    return argv


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _image_present(image: str) -> bool:
    if not _docker_available():
        return False
    try:
        r = subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, timeout=15)
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def preflight(image: str = DEFAULT_IMAGE, synth: str = "estimate") -> Dict:
    """Check prerequisites. Returns {ok, checks:[{name, ok, detail}], blocking}."""
    checks = []

    docker_ok = _docker_available()
    checks.append({"name": "docker", "ok": docker_ok,
                   "detail": "docker CLI found" if docker_ok else "docker not on PATH"})

    img_ok = _image_present(image) if docker_ok else False
    checks.append({"name": "image", "ok": img_ok,
                   "detail": (f"{image} present" if img_ok
                              else f"{image} not pulled (docker pull {image})")})

    xpath = os.environ.get("FINN_XILINX_PATH")
    # Vivado is required for ip/bitfile, optional for estimate
    needs_vivado = synth in ("ip", "bitfile")
    xilinx_ok = bool(xpath) and Path(xpath).exists()
    checks.append({
        "name": "xilinx_tools",
        "ok": xilinx_ok or not needs_vivado,
        "detail": (f"FINN_XILINX_PATH={xpath}" if xilinx_ok
                   else ("FINN_XILINX_PATH not set — required for "
                         f"synth='{synth}'" if needs_vivado
                         else "no Vivado (fine for synth='estimate')")),
    })

    blocking = [c["name"] for c in checks if not c["ok"]]
    return {"ok": not blocking, "checks": checks, "blocking": blocking,
            "command": " ".join(docker_command("BUNDLE", image))}
