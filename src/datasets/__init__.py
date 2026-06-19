"""Datasets: a name-keyed factory with injectable preprocessing.

    from src.datasets import build_dataset, list_datasets
    ds = build_dataset("cifar10", config, transform=optional_transform)
    train_loader, test_loader = ds.get_train_loader(), ds.get_test_loader()

Registered out of the box: cifar10, mnist, fashion_mnist, imagenet/imagefolder
(also .zip archives), openml/jsc (tabular), yolo/coco/detection.
"""

from .base import BaseDataset
# importing the factory triggers best-effort registration of the built-in
# dataset modules (vision/tabular/detection), each guarded so a missing optional
# dependency doesn't break the whole package.
from .factory import build_dataset, list_datasets, register, register_dataset

__all__ = [
    "BaseDataset",
    "build_dataset",
    "list_datasets",
    "register",
    "register_dataset",
]
