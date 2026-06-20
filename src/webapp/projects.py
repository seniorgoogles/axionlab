"""SQLite-backed Projects: the backbone the GUI organizes work around.

A *project* bundles the artifacts of one experiment/effort. Each artifact has a
`kind` so the GUI can group them:

  - model          : a (float) base model           (yaml + optional .pth)
  - quant_model    : a quantized variant            (.pth and/or QONNX .onnx)
  - training_data  : a dataset assigned to the project
  - hardware_model : a hardware/target description  (fpga_part / board)

Artifacts can point back to the artifact they were derived from (`parent_id`),
e.g. a quant_model -> its base model, so the project keeps the lineage that the
transformation tracking will build on later.

Pure stdlib (sqlite3 + json) -- no extra dependency, same style as
src/tracking/experiment_db.py.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# the four artifact types a project can contain
ARTIFACT_KINDS = ("model", "quant_model", "training_data", "hardware_model")


class ProjectDB:
    def __init__(self, db_path: str = "projects.db"):
        self.db_path = db_path
        parent = Path(db_path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                created TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT,
                meta TEXT,                       -- JSON blob (fpga_part, quantize, ...)
                parent_id INTEGER,               -- artifact this was derived from
                created TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES artifacts(id) ON DELETE SET NULL
            );
            """
        )
        self.conn.commit()

    # ---- projects --------------------------------------------------------
    def create_project(self, name: str, description: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO projects (name, description, created) VALUES (?, ?, ?)",
            (name, description, datetime.now().isoformat()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_projects(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["artifact_count"] = self.conn.execute(
                "SELECT COUNT(*) AS n FROM artifacts WHERE project_id = ?", (r["id"],)
            ).fetchone()["n"]
            out.append(d)
        return out

    def get_project(self, project_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        if row is None:
            return None
        proj = dict(row)
        # group artifacts by kind so the GUI can render one section per type
        proj["artifacts"] = {k: [] for k in ARTIFACT_KINDS}
        for a in self.list_artifacts(project_id):
            proj["artifacts"].setdefault(a["kind"], []).append(a)
        return proj

    def delete_project(self, project_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ---- artifacts -------------------------------------------------------
    def add_artifact(self, project_id: int, kind: str, name: str,
                     path: str = "", meta: Optional[dict] = None,
                     parent_id: Optional[int] = None) -> int:
        if kind not in ARTIFACT_KINDS:
            raise ValueError(f"unknown artifact kind '{kind}', expected one of {ARTIFACT_KINDS}")
        if self.conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
            raise ValueError(f"no project with id {project_id}")
        cur = self.conn.execute(
            "INSERT INTO artifacts (project_id, kind, name, path, meta, parent_id, created) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project_id, kind, name, path, json.dumps(meta or {}), parent_id,
             datetime.now().isoformat()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_artifacts(self, project_id: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM artifacts WHERE project_id = ? ORDER BY id", (project_id,)
        ).fetchall()
        return [self._artifact_row(r) for r in rows]

    def delete_artifact(self, artifact_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        self.conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _artifact_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
        except (TypeError, json.JSONDecodeError):
            d["meta"] = {}
        return d

    def close(self) -> None:
        self.conn.close()
