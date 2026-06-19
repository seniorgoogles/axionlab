"""Rich live training dashboard as a Runner callback.

A terminal "GUI" that updates in place: one row per epoch with phase, losses,
accuracy and a *best marker. The phase name encodes the bit width (e.g. 'w4a4'),
so the current precision is visible live.

    from src.ui import RichDashboardCallback
    runner.train(plan=[...], callbacks=[RichDashboardCallback("JSC QAT")])

Needs `rich` (pip install rich). It disables the per-batch tqdm bar to avoid
fighting rich.Live for the terminal.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class RichDashboardCallback:
    def __init__(self, title: str = "training"):
        self.title = title
        self._rows: List[Tuple] = []        # (phase, epoch, train_loss, val_loss, val_acc, best)
        self._phase: Optional[str] = None
        self._best: Optional[float] = None
        self._live = None

    # ---- Runner hooks ----------------------------------------------------
    def on_phase_start(self, runner, phase) -> None:
        runner.show_progress = False         # rich.Live owns the terminal; no tqdm
        self._phase = phase.name
        self._best = None                    # best resets per phase (arch may change)
        if self._live is None:
            from rich.live import Live
            self._live = Live(self._render(), refresh_per_second=8)
            self._live.start()
        self._refresh()

    def on_epoch_end(self, runner, epoch: int, metrics: dict) -> None:
        val_acc = metrics.get("accuracy")
        is_best = val_acc is not None and (self._best is None or val_acc > self._best)
        if is_best:
            self._best = val_acc
        self._rows.append((self._phase, epoch, metrics.get("train_loss"),
                           metrics.get("loss"), val_acc, is_best))
        self._refresh()

    def on_train_end(self, runner, history) -> None:
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None

    # ---- rendering -------------------------------------------------------
    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    @staticmethod
    def _fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "-"

    def _render(self):
        from rich.table import Table

        table = Table(title=self.title, expand=False)
        table.add_column("phase")
        table.add_column("epoch", justify="right")
        table.add_column("train_loss", justify="right")
        table.add_column("val_loss", justify="right")
        table.add_column("val_acc", justify="right")
        for phase, epoch, tl, vl, va, best in self._rows[-20:]:
            acc = self._fmt(va) + (" ★" if best else "")
            style = "bold green" if best else None
            table.add_row(str(phase), str(epoch), self._fmt(tl), self._fmt(vl), acc, style=style)
        return table
