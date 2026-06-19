"""FastAPI controller: serve the dashboard + a small JSON API over JobManager.

Run it with:  ./run_gui.sh           (or: uvicorn src.webapp.server:app --reload)
Then open http://127.0.0.1:8000

Commands are built to call the existing CLI (`python run.py ...`) so the GUI is a
thin wrapper -- exactly the commands you'd type by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.webapp.jobs import JobManager
from src.webapp.runs import list_runs

REPO = str(Path(__file__).resolve().parents[2])     # .../axionlab
STATIC = Path(__file__).parent / "static"
PY = sys.executable

app = FastAPI(title="axionlab control")
jm = JobManager(REPO)


# ---- discovery -------------------------------------------------------------
def _yamls(subdir: str) -> List[str]:
    base = Path(REPO) / subdir
    return sorted(str(p.relative_to(REPO)) for p in base.rglob("*.yaml")) if base.exists() else []


@app.get("/api/config")
def config():
    return {
        "experiments": _yamls("configs/experiments"),
        "models": _yamls("configs/models"),
    }


# ---- start jobs ------------------------------------------------------------
class TrainReq(BaseModel):
    exp: str
    epochs: Optional[int] = None
    lr: Optional[float] = None
    device: Optional[str] = None


class QatReq(BaseModel):
    exp: str
    warmup_epochs: int = 0
    schedule: str = "w8a8:6,w4a4:6"
    min_acc: Optional[float] = None
    device: Optional[str] = None


class ExportReq(BaseModel):
    model: str
    weights: str
    quantize: str = "w4a4"
    input_shape: str = "3,32,32"
    input_quant: int = 8
    fpga_part: str = "xck26-sfvc784-2LV-c"
    out: str = "bundle/model.qonnx.onnx"
    bundle: str = "bundle"
    synth: str = "estimate"


class FinnReq(BaseModel):
    bundle: str                       # cwd to run in
    command: str = "./run-docker.sh python build.py"


def _started(job) -> JSONResponse:
    return JSONResponse({"id": job.id, "cmd": " ".join(job.cmd)})


@app.post("/api/start/train")
def start_train(r: TrainReq):
    cmd = [PY, "run.py", "train", "--exp", r.exp]
    if r.epochs is not None:
        cmd += ["--epochs", str(r.epochs)]
    if r.lr is not None:
        cmd += ["--lr", str(r.lr)]
    if r.device:
        cmd += ["--device", r.device]
    return _started(jm.start("train", cmd))


@app.post("/api/start/qat")
def start_qat(r: QatReq):
    cmd = [PY, "run.py", "qat", "--exp", r.exp,
           "--warmup-epochs", str(r.warmup_epochs), "--schedule", r.schedule]
    if r.min_acc is not None:
        cmd += ["--min-acc", str(r.min_acc)]
    if r.device:
        cmd += ["--device", r.device]
    return _started(jm.start("qat", cmd))


@app.post("/api/start/export")
def start_export(r: ExportReq):
    cmd = [PY, "run.py", "export", "--model", r.model, "--weights", r.weights,
           "--quantize", r.quantize, "--input-shape", r.input_shape,
           "--input-quant", str(r.input_quant), "--fpga-part", r.fpga_part,
           "--out", r.out, "--bundle", r.bundle, "--synth", r.synth]
    return _started(jm.start("export", cmd))


@app.post("/api/start/finn")
def start_finn(r: FinnReq):
    # FINN runs in its docker image; run the given command inside the bundle dir.
    cwd = r.bundle if Path(r.bundle).is_absolute() else str(Path(REPO) / r.bundle)
    return _started(jm.start("finn", r.command.split(), cwd=cwd))


# ---- monitor ---------------------------------------------------------------
@app.post("/api/stop/{job_id}")
def stop(job_id: int):
    return {"stopped": jm.stop(job_id)}


@app.get("/api/jobs")
def jobs():
    return jm.list()


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: int, offset: int = 0):
    return jm.log_since(job_id, offset)


@app.get("/api/runs")
def runs():
    return list_runs(REPO)


# ---- UI --------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
