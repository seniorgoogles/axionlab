"""SQLite-backed experiment tracking for reproducibility.

Stores, in one database, everything needed to reproduce a run:
  - experiments: name + the model yaml + the experiment yaml + timestamp
  - metrics:     per-epoch loss/accuracy per split (train / val / test)
  - weights:     the trained checkpoints (best/last) as BLOBs

So any past result can be looked up and its exact model+weights reloaded.
Pure stdlib (sqlite3) -- no extra dependency.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class ExperimentDB:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        parent = Path(db_path).parent          # create the folder so sqlite can open the file
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_yaml TEXT,
                experiment_yaml TEXT,
                created TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
                exp_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                split TEXT NOT NULL,
                loss REAL,
                accuracy REAL,
                FOREIGN KEY (exp_id) REFERENCES experiments(id)
            );
            CREATE TABLE IF NOT EXISTS weights (
                exp_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                data BLOB NOT NULL,
                saved TEXT NOT NULL,
                PRIMARY KEY (exp_id, kind),
                FOREIGN KEY (exp_id) REFERENCES experiments(id)
            );
            """
        )
        self.conn.commit()

    # ---- writes ----------------------------------------------------------
    def start_experiment(self, name: str, model_yaml: str = "", experiment_yaml: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO experiments (name, model_yaml, experiment_yaml, created) VALUES (?, ?, ?, ?)",
            (name, model_yaml, experiment_yaml, datetime.now().isoformat()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_metric(self, exp_id: int, epoch: int, split: str,
                   loss: Optional[float] = None, accuracy: Optional[float] = None) -> None:
        self.conn.execute(
            "INSERT INTO metrics (exp_id, epoch, split, loss, accuracy) VALUES (?, ?, ?, ?, ?)",
            (exp_id, epoch, split, loss, accuracy),
        )
        self.conn.commit()

    def save_weights(self, exp_id: int, kind: str, path: str) -> None:
        """Store a checkpoint file's bytes (kind = 'best' / 'last')."""
        data = Path(path).read_bytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO weights (exp_id, kind, data, saved) VALUES (?, ?, ?, ?)",
            (exp_id, kind, data, datetime.now().isoformat()),
        )
        self.conn.commit()

    # ---- reads -----------------------------------------------------------
    def get_experiment(self, exp_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()

    def get_metrics(self, exp_id: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM metrics WHERE exp_id = ? ORDER BY epoch, split", (exp_id,)
        ).fetchall()

    def load_weights(self, exp_id: int, kind: str = "best") -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT data FROM weights WHERE exp_id = ? AND kind = ?", (exp_id, kind)
        ).fetchone()
        return bytes(row["data"]) if row else None

    def list_experiments(self) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()

    def close(self) -> None:
        self.conn.close()


class DBTrackerCallback:
    """Runner callback that records metrics + weights into an ExperimentDB.

    Attach via Runner.train(callbacks=[DBTrackerCallback(db, name, model_yaml, exp_yaml)]).
    """

    def __init__(self, db: ExperimentDB, name: str, model_yaml: str = "", experiment_yaml: str = ""):
        self.db = db
        self.name = name
        self.model_yaml = model_yaml
        self.experiment_yaml = experiment_yaml
        self.exp_id: Optional[int] = None

    def _ensure_started(self) -> None:
        if self.exp_id is None:
            self.exp_id = self.db.start_experiment(self.name, self.model_yaml, self.experiment_yaml)

    def on_phase_start(self, runner, phase) -> None:
        self._ensure_started()

    def on_epoch_end(self, runner, epoch: int, metrics: dict) -> None:
        self._ensure_started()
        if "train_loss" in metrics:
            self.db.log_metric(self.exp_id, epoch, "train", loss=metrics.get("train_loss"))
        if "loss" in metrics:  # validation pass happened this epoch
            self.db.log_metric(self.exp_id, epoch, "val",
                               loss=metrics.get("loss"), accuracy=metrics.get("accuracy"))

    def on_train_end(self, runner, history) -> None:
        self._ensure_started()
        save_dir = Path(runner.config.save_dir)
        for kind in ("best", "last"):
            f = save_dir / f"{kind}.pth"
            if f.exists():
                self.db.save_weights(self.exp_id, kind, str(f))
