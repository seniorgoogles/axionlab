"""Datasets module for data loading and preprocessing.

This module provides dataset classes and utilities for loading various datasets.

Components:
- BaseDataset: Abstract base class for all datasets
- DatasetBuilder: Factory for building dataset instances
- DatasetRegistry: Registry for dataset types
- Individual dataset implementations (MNIST, FashionMNIST, ImageNet, etc.)
"""

from .base import BaseDataset
from .dataset_builder import DatasetBuilder, DatasetRegistry
from .mnist import Mnist
from .fashionMnist import FashionMnist
from .cifar10 import Cifar10
from .imagenet import ImageNet

__all__ = [
    "BaseDataset",
    "DatasetBuilder",
    "DatasetRegistry",
    "Mnist",
    "FashionMnist",
    "Cifar10",
    "ImageNet",
]
