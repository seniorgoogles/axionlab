"""SQLite-backed transformation tracking: the lineage from model to hardware.

The path to hardware is a chain of steps, each turning one artifact into the next:

    float --qat--> quant --export--> qonnx --finn:streamline--> ... --> bitfile

We record one row per *step* so the chain is reproducible and debuggable:
  - which step ran, with which params, on which input -> which output
  - whether it succeeded (status + error), so a failure is pinned to its step
  - an optional ONNX snapshot of the step's output, for the interactive viewer

Steps are grouped by `run_id` and ordered by `seq`, so a chain can be replayed
or restarted from the first failed step instead of from scratch.

Pure stdlib (sqlite3 + json); shares the projects.db file but needs no other
table to exist, so it is testable standalone.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

STATUSES = ("running", "ok", "failed")


class TransformDB:
    def __init__(self, db_path: str = "projects.db"):
        self.db_path = db_path
        parent = Path(db_path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS transforms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,              -- loose link to a project (no FK)
                run_id TEXT NOT NULL,            -- groups the steps of one chain
                seq INTEGER NOT NULL,            -- order within the run
                step TEXT NOT NULL,              -- 'qat' | 'export' | 'finn:streamline' | ...
                input TEXT,                      -- input artifact (path/description)
                output TEXT,                     -- output artifact (path/description)
                params TEXT,                     -- JSON
                status TEXT NOT NULL DEFAULT 'ok',
                error TEXT,                      -- failure message, if any
                snapshot TEXT,                   -- ONNX snapshot path for this step
                created TEXT NOT NULL,
                ended TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_transforms_run ON transforms(run_id, seq);
            """
        )
        self.conn.commit()

    # ---- writes ----------------------------------------------------------
    def start_step(self, run_id: str, seq: int, step: str, project_id: Optional[int] = None,
                   input: str = "", params: Optional[dict] = None) -> int:
        """Record a step that has begun (status 'running'). Returns its id."""
        cur = self.conn.execute(
            "INSERT INTO transforms (project_id, run_id, seq, step, input, params, status, created) "
            "VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
            (project_id, run_id, seq, step, input, json.dumps(params or {}),
             datetime.now().isoformat()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_step(self, step_id: int, status: str = "ok", output: str = "",
                    error: str = "", snapshot: str = "") -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status '{status}', expected one of {STATUSES}")
        self.conn.execute(
            "UPDATE transforms SET status = ?, output = ?, error = ?, snapshot = ?, ended = ? "
            "WHERE id = ?",
            (status, output, error, snapshot, datetime.now().isoformat(), step_id),
        )
        self.conn.commit()

    def record_step(self, run_id: str, seq: int, step: str, status: str = "ok",
                    project_id: Optional[int] = None, input: str = "", output: str = "",
                    params: Optional[dict] = None, error: str = "", snapshot: str = "") -> int:
        """Record a step that already completed (one shot)."""
        if status not in STATUSES:
            raise ValueError(f"unknown status '{status}', expected one of {STATUSES}")
        now = datetime.now().isoformat()
        cur = self.conn.execute(
            "INSERT INTO transforms (project_id, run_id, seq, step, input, output, params, "
            "status, error, snapshot, created, ended) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, run_id, seq, step, input, output, json.dumps(params or {}),
             status, error, snapshot, now, now),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # ---- reads -----------------------------------------------------------
    def get_run(self, run_id: str) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM transforms WHERE run_id = ? ORDER BY seq, id", (run_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def list_runs(self, project_id: Optional[int] = None) -> List[dict]:
        """One summary row per run_id (latest first)."""
        if project_id is None:
            rows = self.conn.execute("SELECT * FROM transforms ORDER BY id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM transforms WHERE project_id = ? ORDER BY id", (project_id,)
            ).fetchall()
        runs: dict = {}
        for r in rows:
            d = self._row(r)
            run = runs.setdefault(d["run_id"], {
                "run_id": d["run_id"], "project_id": d["project_id"],
                "steps": 0, "failed": 0, "status": "ok", "created": d["created"],
            })
            run["steps"] += 1
            if d["status"] == "failed":
                run["failed"] += 1
                run["status"] = "failed"
            elif d["status"] == "running" and run["status"] != "failed":
                run["status"] = "running"
        return sorted(runs.values(), key=lambda x: x["created"] or "", reverse=True)

    def first_failed(self, run_id: str) -> Optional[dict]:
        """The earliest failed step in a run -- the point to restart/repair from."""
        for step in self.get_run(run_id):
            if step["status"] == "failed":
                return step
        return None

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["params"] = json.loads(d["params"]) if d["params"] else {}
        except (TypeError, json.JSONDecodeError):
            d["params"] = {}
        return d

    def close(self) -> None:
        self.conn.close()
