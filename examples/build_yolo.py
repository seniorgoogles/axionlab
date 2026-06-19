#!/usr/bin/env python
"""Build YOLOv8n from the axionlab graph config and run a dummy forward.

Run from the repo root:
    python examples/build_yolo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import torch

from src.core.build import build_yolo_from_config


def main():
    model = build_yolo_from_config("configs/models/yolov8n.yaml")
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 640, 640))
    # Detect returns a list of feature maps (training mode) or decoded preds (eval).
    if isinstance(out, (list, tuple)):
        print("Detect outputs:", [tuple(o.shape) for o in out])
    else:
        print("output:", tuple(out.shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")


if __name__ == "__main__":
    main()
