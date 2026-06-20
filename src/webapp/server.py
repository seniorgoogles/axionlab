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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.webapp import expbuilder, finn_docker
from src.webapp.jobs import JobManager
from src.webapp.projects import ARTIFACT_KINDS, ProjectDB
from src.webapp.runs import list_runs
from src.webapp.transforms import TransformDB

REPO = str(Path(__file__).resolve().parents[2])     # .../axionlab
STATIC = Path(__file__).parent / "static"
PY = sys.executable
_DB = str(Path(REPO) / "projects.db")

app = FastAPI(title="axionlab control")
jm = JobManager(REPO)
pdb = ProjectDB(_DB)
tdb = TransformDB(_DB)


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
    # make the default './run-docker.sh ...' work by dropping the script into the bundle
    if r.command.strip().startswith("./run-docker.sh") and Path(cwd).is_dir():
        finn_docker.ensure_run_docker(cwd)
    return _started(jm.start("finn", r.command.split(), cwd=cwd))


@app.get("/api/finn/preflight")
def finn_preflight(synth: str = "estimate"):
    return finn_docker.preflight(synth=synth)


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


# ---- projects --------------------------------------------------------------
class ProjectReq(BaseModel):
    name: str
    description: str = ""


class ArtifactReq(BaseModel):
    kind: str                          # model | quant_model | training_data | hardware_model
    name: str
    path: str = ""
    meta: dict = {}
    parent_id: Optional[int] = None


@app.get("/api/artifact-kinds")
def artifact_kinds():
    return list(ARTIFACT_KINDS)


@app.get("/api/projects")
def list_projects():
    return pdb.list_projects()


@app.post("/api/projects")
def create_project(r: ProjectReq):
    if not r.name.strip():
        raise HTTPException(400, "project name is required")
    return {"id": pdb.create_project(r.name.strip(), r.description)}


@app.get("/api/projects/{project_id}")
def get_project(project_id: int):
    proj = pdb.get_project(project_id)
    if proj is None:
        raise HTTPException(404, f"no project with id {project_id}")
    return proj


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: int):
    if not pdb.delete_project(project_id):
        raise HTTPException(404, f"no project with id {project_id}")
    return {"deleted": project_id}


@app.post("/api/projects/{project_id}/artifacts")
def add_artifact(project_id: int, r: ArtifactReq):
    try:
        aid = pdb.add_artifact(project_id, r.kind, r.name, r.path, r.meta, r.parent_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": aid}


@app.delete("/api/projects/{project_id}/artifacts/{artifact_id}")
def delete_artifact(project_id: int, artifact_id: int):
    if not pdb.delete_artifact(artifact_id):
        raise HTTPException(404, f"no artifact with id {artifact_id}")
    return {"deleted": artifact_id}


# ---- transforms (lineage) --------------------------------------------------
class StepReq(BaseModel):
    run_id: str
    seq: int
    step: str
    status: str = "ok"
    project_id: Optional[int] = None
    input: str = ""
    output: str = ""
    params: dict = {}
    error: str = ""
    snapshot: str = ""


@app.get("/api/transforms")
def transforms(project_id: Optional[int] = None):
    return tdb.list_runs(project_id)


@app.get("/api/transforms/{run_id}")
def transform_run(run_id: str):
    steps = tdb.get_run(run_id)
    if not steps:
        raise HTTPException(404, f"no transform run '{run_id}'")
    return {"run_id": run_id, "steps": steps, "first_failed": tdb.first_failed(run_id)}


@app.post("/api/transforms")
def record_step(r: StepReq):
    try:
        sid = tdb.record_step(r.run_id, r.seq, r.step, r.status, r.project_id,
                              r.input, r.output, r.params, r.error, r.snapshot)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": sid}


# ---- experiment builder ----------------------------------------------------
class ExpSpec(BaseModel):
    name: str
    model: str                          # path to a model (network) yaml
    dataset: str
    epochs: int = 30
    learning_rate: float = 0.01
    batch_size: List[int] = [128, 256]
    num_workers: int = 4
    distributed: bool = False
    train_path: Optional[str] = None
    output_dir: Optional[str] = None
    best_metric_name: str = "accuracy"
    best_metric_mode: str = "max"
    validate_every: int = 1
    # save-only fields
    path: Optional[str] = None          # where to write (relative to repo)
    overwrite: bool = False


@app.get("/api/datasets")
def datasets():
    return expbuilder.list_datasets()


@app.get("/api/model-content")
def model_content(path: str):
    """Open a network description (model yaml) so the GUI can show it."""
    try:
        return {"path": path, "content": expbuilder.read_model(REPO, path)}
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.post("/api/experiments/preview")
def preview_experiment(s: ExpSpec):
    try:
        cfg = expbuilder.build_exp_config(s.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"yaml": expbuilder.to_yaml(cfg),
            "default_path": expbuilder.default_path(REPO, cfg["name"])}


@app.post("/api/experiments")
def save_experiment(s: ExpSpec):
    try:
        cfg = expbuilder.build_exp_config(s.model_dump())
        res = expbuilder.save_exp_config(REPO, cfg, s.path, s.overwrite)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileExistsError as e:
        raise HTTPException(409, f"{e} already exists (set overwrite to replace)")
    return res


# ---- UI --------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
