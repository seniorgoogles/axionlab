"""Progressive QAT bit-width schedule with rollback.

Lower the bit width in stages (e.g. w8a8 -> w4a4 -> w2a2), each a fine-tuning
phase. Weights and activations are scheduled INDEPENDENTLY (each stage names its
own weight_bit_width and act_bit_width), so you can lower one at a time, e.g.
w8a8 -> w4a8 -> w4a4. If a stage cannot reach its `min_accuracy`, the model is
rolled back to the previous (good) stage and the schedule stops.

    from src.core import Runner
    from src.core.qat_schedule import QatStage, run_qat_schedule

    runner = Runner.from_experiment("...yaml")
    runner.train(plan=[Phase("float_warmup", 8, lr=1e-3)])      # optional float warmup
    run_qat_schedule(runner, [
        QatStage("w8a8", 8, 8, epochs=6, lr=5e-4, min_accuracy=0.74),
        QatStage("w4a8", 4, 8, epochs=6, lr=3e-4, min_accuracy=0.73),  # weights only
        QatStage("w4a4", 4, 4, epochs=6, lr=2e-4, min_accuracy=0.72),  # then activations
        QatStage("w2a2", 2, 2, epochs=8, lr=1e-4, min_accuracy=0.68),
    ])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

from src.core.quantize import set_bit_width
from src.core.runner import Phase, Runner


@dataclass
class QatStage:
    name: str
    weight_bit_width: int
    act_bit_width: int
    epochs: int
    lr: float
    min_accuracy: Optional[float] = None   # roll back + stop if best val metric is below this


def _quantizer(wb: int, ab: int):
    return lambda r: r.replace_model(set_bit_width(r.model, wb, ab))


def run_qat_schedule(runner: Runner, stages: List[QatStage], callbacks: Optional[list] = None) -> dict:
    """Run the stages in order; roll back to the last good model if a stage misses
    its min_accuracy. Returns a summary dict."""
    history = []
    rolled_back = False
    stopped_at = None

    for stage in stages:
        snapshot = copy.deepcopy(runner.model)        # last good model (before this stage)

        hist = runner.train(
            plan=[Phase(stage.name, epochs=stage.epochs, lr=stage.lr,
                        on_start=_quantizer(stage.weight_bit_width, stage.act_bit_width))],
            callbacks=callbacks,
        )
        achieved = max((m for m in hist["val_metric"] if m is not None), default=None)
        history.append({"stage": stage.name, "wb": stage.weight_bit_width,
                        "ab": stage.act_bit_width, "best_metric": achieved})
        stopped_at = stage.name

        if stage.min_accuracy is not None and (achieved is None or achieved < stage.min_accuracy):
            print(f"[qat] stage '{stage.name}' missed target {stage.min_accuracy} "
                  f"(best={achieved}); rolling back to previous bit width and stopping.")
            runner.replace_model(snapshot)
            # make best.pth reflect the rolled-back (last good) model
            runner.ckpt.reset_best()
            runner.ckpt.save_checkpoint(runner.model, "best.pth", epoch=0, is_best=True)
            rolled_back = True
            break

        print(f"[qat] stage '{stage.name}' ok (best={achieved}).")

    return {"history": history, "rolled_back": rolled_back, "stopped_at": stopped_at}
