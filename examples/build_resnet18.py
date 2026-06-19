#!/usr/bin/env python
"""Build ResNet-18 from the layer-list yaml and load pretrained float weights.

Run from the repo root:
    python examples/build_resnet18.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import torch

from src.core.build import build_from_config, load_float_state_dict


def main():
    model = build_from_config("configs/models/resnet18.yaml")
    print(model)

    # load pretrained FLOAT weights into the (quantized) model, by name
    try:
        import torchvision
        load_float_state_dict(model, torchvision.models.resnet18(weights="DEFAULT"))
    except Exception as e:
        print(f"(skipping torchvision weight load: {e})")

    # sanity forward
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 224, 224))
    print("output shape:", tuple(y.shape))   # expect (1, 1000)


if __name__ == "__main__":
    main()
