"""Vision classification datasets, all BaseDataset-based and transform-aware.

- ImageFolderDataset: a folder (or .zip archive, ImageNet-style) of class subdirs.
- CIFAR10 / MNIST / FashionMNIST: torchvision built-ins with default transforms.

All honor an injected `self.transform` (overrides the default) so preprocessing
is configurable per experiment.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import torchvision.datasets as tvd

from src.datasets.base import BaseDataset
from src.datasets.factory import register_dataset
from src.datasets.preprocessing import (
    cifar_transform,
    classification_transform,
    gray_transform,
)


def _maybe_extract(path: str) -> str:
    """If `path` is a .zip, extract it next to itself once and return the dir."""
    if path is None:
        return path
    p = Path(path)
    if p.suffix == ".zip" and p.exists():
        out = p.with_suffix("")
        if not out.exists():
            with zipfile.ZipFile(p) as zf:
                zf.extractall(out)
        return str(out)
    return path


@register_dataset("imagefolder", "imagenet")
class ImageFolderDataset(BaseDataset):
    """Classification from class-subfolder directories (or .zip archives)."""

    def _do_preprocessing(self, image_size: int = 224, **kwargs):
        train_dir = _maybe_extract(self.train_path)
        test_dir = _maybe_extract(self.test_path)
        train_tf = self.transform or classification_transform(image_size, train=True)
        test_tf = self.transform or classification_transform(image_size, train=False)
        return tvd.ImageFolder(train_dir, transform=train_tf), tvd.ImageFolder(test_dir, transform=test_tf)


class _TorchvisionDataset(BaseDataset):
    """Shared base for torchvision built-ins keyed by train/test split flag."""

    tv_cls = None

    def _default_transform(self, train: bool):
        raise NotImplementedError

    def _root(self) -> str:
        return self.train_path or self.test_path or "./tmp/datasets"

    def _do_preprocessing(self, **kwargs):
        root = self._root()
        train_tf = self.transform or self._default_transform(train=True)
        test_tf = self.transform or self._default_transform(train=False)
        return self._load(root, True, train_tf), self._load(root, False, test_tf)

    def _load(self, root: str, train: bool, transform):
        # Reuse a cached dataset instead of re-downloading / re-checking integrity
        # on every restart: try download=False first, only fetch once if missing.
        try:
            return self.tv_cls(root, train=train, download=False, transform=transform)
        except (RuntimeError, FileNotFoundError):
            return self.tv_cls(root, train=train, download=True, transform=transform)


@register_dataset("cifar10")
class Cifar10(_TorchvisionDataset):
    tv_cls = tvd.CIFAR10

    def _default_transform(self, train: bool):
        return cifar_transform(train=train)


@register_dataset("mnist")
class Mnist(_TorchvisionDataset):
    tv_cls = tvd.MNIST

    def _default_transform(self, train: bool):
        return gray_transform()


@register_dataset("fashion_mnist", "fashionmnist")
class FashionMnist(_TorchvisionDataset):
    tv_cls = tvd.FashionMNIST

    def _default_transform(self, train: bool):
        return gray_transform()
