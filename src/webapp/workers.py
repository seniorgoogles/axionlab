"""Distribute jobs across machines.

A *worker* is just another axionlab server instance: it exposes the same job API
(/api/start/..., /api/jobs, ...). The controller keeps a registry of workers with
their capabilities (docker / gpu / vivado) and routes a job to a suitable one --
e.g. a FINN synthesis to a host that has Vivado.

This module is the registry + routing + capability detection (pure/testable). The
actual forwarding is a thin HTTP proxy in server.py using `requests`.

Capabilities are detected without importing torch, so a GUI-only worker can still
advertise what it can do.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# capability -> the job kinds that need it (used for routing)
CAPABILITY_FOR_KIND = {
    "finn": "vivado",      # synthesis/bitfile needs Xilinx tools (estimate is lenient)
}


def detect_capabilities() -> List[str]:
    """What this machine can do, best-effort and torch-free."""
    caps = []
    if shutil.which("docker"):
        caps.append("docker")
    if shutil.which("nvidia-smi"):
        caps.append("gpu")
    xpath = os.environ.get("FINN_XILINX_PATH")
    if xpath and Path(xpath).exists():
        caps.append("vivado")
    return caps


def pick_worker(workers: List[dict], capability: Optional[str]) -> Optional[dict]:
    """First worker that has `capability` (or the first worker if none required)."""
    if not workers:
        return None
    if not capability:
        return workers[0]
    for w in workers:
        if capability in (w.get("capabilities") or []):
            return w
    return None


class WorkerRegistry:
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
            CREATE TABLE IF NOT EXISTS workers (
                name TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                capabilities TEXT,          -- JSON list
                created TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def register(self, name: str, url: str, capabilities: Optional[List[str]] = None) -> None:
        if not name.strip() or not url.strip():
            raise ValueError("worker name and url are required")
        self.conn.execute(
            "INSERT OR REPLACE INTO workers (name, url, capabilities, created) VALUES (?, ?, ?, ?)",
            (name.strip(), url.strip().rstrip("/"), json.dumps(capabilities or []),
             datetime.now().isoformat()),
        )
        self.conn.commit()

    def list(self) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM workers ORDER BY name").fetchall()
        return [self._row(r) for r in rows]

    def get(self, name: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM workers WHERE name = ?", (name,)).fetchone()
        return self._row(r) if r else None

    def remove(self, name: str) -> bool:
        cur = self.conn.execute("DELETE FROM workers WHERE name = ?", (name,))
        self.conn.commit()
        return cur.rowcount > 0

    def route(self, kind: str) -> Optional[dict]:
        """Pick a worker able to run a job of this kind."""
        return pick_worker(self.list(), CAPABILITY_FOR_KIND.get(kind))

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        try:
            d["capabilities"] = json.loads(d["capabilities"]) if d["capabilities"] else []
        except (TypeError, json.JSONDecodeError):
            d["capabilities"] = []
        return d

    def close(self) -> None:
        self.conn.close()
