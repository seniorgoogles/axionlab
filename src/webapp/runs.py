"""Read past runs from the tracking SQLite DB(s) for the runs browser.

Scans the repo for experiments.db files and, for each experiment, returns its
name, timestamp, best val accuracy and last epoch. Read-only and schema-tolerant:
a DB that doesn't match the expected schema is skipped instead of crashing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List


def find_dbs(repo_root: str) -> List[Path]:
    root = Path(repo_root)
    return [p for p in root.rglob("experiments.db") if ".git" not in p.parts]


def list_runs(repo_root: str) -> List[dict]:
    runs: List[dict] = []
    for db in find_dbs(repo_root):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            exps = conn.execute(
                "SELECT id, name, model_yaml, created FROM experiments ORDER BY id DESC"
            ).fetchall()
            for e in exps:
                agg = conn.execute(
                    "SELECT MAX(accuracy) AS best_acc, MAX(epoch) AS last_epoch "
                    "FROM metrics WHERE exp_id = ? AND split = 'val'",
                    (e["id"],),
                ).fetchone()
                runs.append({
                    "db": str(db.relative_to(repo_root)),
                    "id": e["id"],
                    "name": e["name"],
                    "model": e["model_yaml"],
                    "created": e["created"],
                    "best_acc": agg["best_acc"] if agg else None,
                    "last_epoch": agg["last_epoch"] if agg else None,
                })
            conn.close()
        except sqlite3.Error:
            continue
    runs.sort(key=lambda r: r["created"] or "", reverse=True)
    return runs
