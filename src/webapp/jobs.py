"""Subprocess job manager: run CLI commands, capture logs, allow stop.

A Job wraps one `python run.py ...` (or a FINN docker command) running as a
child process. stdout+stderr are streamed line-by-line into a ring buffer so the
UI can poll new lines. Stopping kills the whole process group (so child
processes started by the command die too).

This same interface is what the future remote worker agent will expose, so the
controller can treat local and remote jobs uniformly.
"""

from __future__ import annotations

import itertools
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

_ids = itertools.count(1)


@dataclass
class Job:
    id: int
    kind: str                      # train | qat | export | finn
    cmd: List[str]
    cwd: str
    status: str = "running"        # running | done | failed | stopped
    returncode: Optional[int] = None
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=10000))
    proc: Optional[subprocess.Popen] = None

    def summary(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "cmd": " ".join(self.cmd),
            "cwd": self.cwd, "status": self.status, "returncode": self.returncode,
            "started": self.started, "ended": self.ended, "loglen": len(self.log),
        }


class JobManager:
    """Owns all running/finished jobs for the controller process."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.jobs: Dict[int, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, cmd: List[str], cwd: Optional[str] = None) -> Job:
        job = Job(id=next(_ids), kind=kind, cmd=cmd, cwd=cwd or self.repo_root)
        # start_new_session so we can kill the whole process group on stop
        job.proc = subprocess.Popen(
            cmd, cwd=job.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def _pump(self, job: Job) -> None:
        assert job.proc and job.proc.stdout
        for line in job.proc.stdout:
            job.log.append(line.rstrip("\n"))
        job.proc.wait()
        job.returncode = job.proc.returncode
        job.ended = time.time()
        if job.status == "running":      # don't overwrite an explicit 'stopped'
            job.status = "done" if job.returncode == 0 else "failed"

    def stop(self, job_id: int) -> bool:
        job = self.jobs.get(job_id)
        if not job or not job.proc or job.status != "running":
            return False
        job.status = "stopped"
        try:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        return True

    def log_since(self, job_id: int, offset: int) -> dict:
        """Return log lines from `offset` onward (offset = #lines already seen)."""
        job = self.jobs.get(job_id)
        if not job:
            return {"lines": [], "next_offset": offset, "status": "missing"}
        lines = list(job.log)
        new = lines[offset:] if offset < len(lines) else []
        return {"lines": new, "next_offset": len(lines), "status": job.status}

    def list(self) -> List[dict]:
        with self._lock:
            return [j.summary() for j in sorted(self.jobs.values(), key=lambda x: -x.id)]
