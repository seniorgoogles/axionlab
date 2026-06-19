"""Default preprocessing/transforms per modality.

Each dataset ships a sensible default so it works out of the box; an injected
`transform` (via the factory) overrides it. Kept here so transforms are reusable
and not duplicated across dataset classes (DRY).
"""

from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classification_transform(image_size: int = 224, train: bool = False,
                             mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Standard RGB classification pipeline. Train uses light augmentation."""
    if train:
        steps = [T.RandomResizedCrop(image_size), T.RandomHorizontalFlip()]
    else:
        steps = [T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size)]
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


def gray_transform(normalize=((0.5,), (0.5,))):
    """MNIST/FashionMNIST style: single-channel, normalized."""
    return T.Compose([T.ToTensor(), T.Normalize(*normalize)])


def cifar_transform(train: bool = False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    steps = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(steps)


def detection_transform(image_size: int = 640):
    """Minimal detection image transform (resize + tensor). Boxes are handled by
    the detection dataset, not here."""
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
