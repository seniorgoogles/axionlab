#!/usr/bin/env python
"""Progressive QAT on JSC with a live Rich dashboard.

float warmup -> w8a8 -> w4a8 (weights only) -> w4a4 -> w2a2, each a fine-tuning
stage. A stage that misses its min_accuracy rolls the model back to the previous
bit width and stops. Progress shows in a live terminal table.

    python examples/jsc_qat_progressive.py

Needs: scikit-learn (OpenML JSC) and rich (pip install rich scikit-learn).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.core import Phase, Runner
from src.core.qat_schedule import QatStage, run_qat_schedule
from src.ui import RichDashboardCallback


def main():
    runner = Runner.from_experiment("configs/experiments/jsc/jsc_2l_qat.yaml")
    dash = RichDashboardCallback("JSC progressive QAT")

    # 1) float warmup
    runner.train(plan=[Phase("float_warmup", epochs=8, lr=1e-3)], callbacks=[dash])

    # 2) progressive quantization (weights and activations scheduled independently)
    result = run_qat_schedule(runner, [
        QatStage("w8a8", weight_bit_width=8, act_bit_width=8, epochs=6, lr=5e-4, min_accuracy=0.72),
        QatStage("w4a8", weight_bit_width=4, act_bit_width=8, epochs=6, lr=3e-4, min_accuracy=0.70),
        QatStage("w4a4", weight_bit_width=4, act_bit_width=4, epochs=6, lr=2e-4, min_accuracy=0.68),
        QatStage("w2a2", weight_bit_width=2, act_bit_width=2, epochs=8, lr=1e-4, min_accuracy=0.60),
    ], callbacks=[dash])

    print("schedule result:", result)
    print("eval (best.pth):", runner.evaluate())


if __name__ == "__main__":
    main()
