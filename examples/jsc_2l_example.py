#!/usr/bin/env python
"""Train JSC-2L via the Runner.

    python examples/jsc_2l_example.py

Builds the net from configs/models/jsc_2l.yaml, loads the JSC dataset through the
factory, and trains -> best.pth / last.pth in outputs/jsc_2l/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.core import Runner


def main():
    runner = Runner.from_experiment("configs/experiments/jsc/jsc_2l_runner.yaml")
    runner.train()
    print("eval (best.pth):", runner.evaluate())


if __name__ == "__main__":
    main()
