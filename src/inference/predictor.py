"""Backend-agnostic inference predictors.

The GUI (and any caller) depends only on the `Predictor` interface, so a PyTorch
model and (later) a FINN-deployed accelerator are interchangeable backends
(Dependency Inversion). Classification is implemented here; a detection or FINN
predictor implements the same `predict()` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Protocol


@dataclass
class Prediction:
    label: str
    score: float
    raw: Any = None


class Predictor(Protocol):
    def predict(self, image) -> Prediction:  # image: PIL.Image
        ...


class TorchClassifier:
    """PyTorch classification backend."""

    def __init__(self, model, class_names: Optional[List[str]] = None,
                 device: str = "cpu", image_size: int = 224, transform=None):
        import torch
        from src.datasets.preprocessing import classification_transform

        self._torch = torch
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.class_names = class_names
        self.transform = transform or classification_transform(image_size, train=False)

    def predict(self, image) -> Prediction:
        x = self.transform(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits = self.model(x)
        probs = self._torch.softmax(logits, dim=1)[0]
        score, idx = probs.max(0)
        idx = int(idx)
        label = self.class_names[idx] if self.class_names and idx < len(self.class_names) else str(idx)
        return Prediction(label=label, score=float(score), raw=probs.cpu().numpy())


def load_classifier(model_yaml: str, checkpoint: str, class_names=None,
                    device: str = "cpu", image_size: int = 224) -> TorchClassifier:
    """Build the net from its yaml, load weights, return a ready predictor."""
    import torch

    from src.core import build_network
    model = build_network(model_yaml)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return TorchClassifier(model, class_names=class_names, device=device, image_size=image_size)
