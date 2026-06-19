"""Spec for the inference predictor (task #17). GUI itself is not unit-tested."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")


def test_torch_classifier_predicts_label():
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    from src.inference import TorchClassifier

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    clf = TorchClassifier(model, class_names=["cat", "dog"], transform=T.ToTensor())

    pred = clf.predict(Image.new("RGB", (4, 4)))
    assert pred.label in {"cat", "dog"}
    assert 0.0 <= pred.score <= 1.0
