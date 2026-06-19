#!/usr/bin/env python
"""QAT run on JSC: float warmup -> quantize -> fine-tune, tracked in a DB.

    python examples/jsc_qat.py

Phase recipe:
  1) train the float MLP for a few epochs (warmup),
  2) at the QAT phase, swap layers to Brevitas quant equivalents (weights carried
     over) and fine-tune at a lower lr.
Writes best.pth / last.pth + metrics_history.json in outputs/jsc_2l_qat/, and
records metrics + weights + yamls into outputs/jsc_2l_qat/experiments.db.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.core import Phase, Runner, quantize_model


def main():
    runner = Runner.from_experiment("configs/experiments/jsc/jsc_2l_qat.yaml")

    runner.train(plan=[
        Phase("float_warmup", epochs=10, lr=1e-3),
        Phase("qat_finetune", epochs=20, lr=1e-4,
              on_start=lambda r: r.replace_model(quantize_model(r.model, weight_bit_width=4, act_bit_width=4))),
    ])

    print("eval (best.pth):", runner.evaluate())


if __name__ == "__main__":
    main()
