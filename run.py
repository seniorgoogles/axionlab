#!/usr/bin/env python
"""axionlab CLI entry point.

    python run.py train --exp configs/experiments/jsc/jsc_2l_qat.yaml --epochs 10 --dashboard
    python run.py qat   --exp configs/experiments/jsc/jsc_2l_qat.yaml --warmup-epochs 5 \
                        --schedule w8a8:6,w4a4:6,w2a2:8 --min-acc 0.7 --dashboard
    python run.py infer --exp configs/models/resnet18.yaml --weights best.pth --source webcam --gui
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # repo root on path

from src.cli import main

if __name__ == "__main__":
    main()
