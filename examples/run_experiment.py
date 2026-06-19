#!/usr/bin/env python
"""Run a full experiment (train -> best/last checkpoints -> evaluate) via the Runner.

    python examples/run_experiment.py configs/experiments/imagenet/resnet18.yaml
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.core import Runner


def main():
    experiment = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/imagenet/resnet18.yaml"
    runner = Runner.from_experiment(experiment)

    runner.train()                       # writes best.pth + last.pth + metrics_history.json
    metrics = runner.evaluate()          # loads best.pth, runs the test/val loader
    print("eval (best.pth):", metrics)


if __name__ == "__main__":
    main()
