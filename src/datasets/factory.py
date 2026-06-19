"""Name-keyed dataset factory with injectable preprocessing.

Register a dataset under one or more names and build it by name from an
experiment config. A `transform` (preprocessing) can be injected at build time;
it is forwarded to the dataset, which should prefer `self.transform` over its
default when set.

    from src.datasets.factory import build_dataset, register_dataset

    @register_dataset("cifar10")
    class Cifar10(BaseDataset):
        ...

    ds = build_dataset("cifar10", config, transform=my_transform)
    train_loader, test_loader = ds.get_train_loader(), ds.get_test_loader()
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional

_REGISTRY: Dict[str, type] = {}


def register_dataset(*names: str) -> Callable[[type], type]:
    """Class decorator: register a BaseDataset subclass under one or more names."""
    def deco(cls: type) -> type:
        for n in names:
            _REGISTRY[n.lower()] = cls
        return cls
    return deco


def register(name: str, cls: type) -> None:
    """Imperative registration (for classes defined elsewhere)."""
    _REGISTRY[name.lower()] = cls


def list_datasets() -> list:
    return sorted(_REGISTRY)


def build_dataset(name: str, config: Dict[str, Any], transform: Optional[Callable] = None,
                  **overrides) -> Any:
    """Instantiate a registered dataset from an experiment config.

    Args:
        name: registered dataset name (case-insensitive).
        config: experiment/dataset config dict (train_path, test_path, batch_size, ...).
        transform: optional preprocessing to inject (only passed if the dataset accepts it).
        overrides: extra kwargs that take precedence over config-derived ones.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {list_datasets()}")
    cls = _REGISTRY[key]

    batch_size = config.get("batch_size", [32, 32])
    if isinstance(batch_size, int):
        batch_size = [batch_size, batch_size]

    kwargs: Dict[str, Any] = {
        "train_path": config.get("train_path"),
        "test_path": config.get("test_path"),
        "batch_size": batch_size,
        "distributed_training": config.get("distributed", False),
        "num_workers": config.get("num_workers", 4),
    }
    # JSC-style datasets take a single root path instead of train/test paths
    if config.get("dataset_root_path"):
        kwargs["dataset_root_path"] = config["dataset_root_path"]
    if config.get("dataset_path"):
        kwargs["dataset_path"] = config["dataset_path"]
    if transform is not None:
        kwargs["transform"] = transform
    kwargs.update(overrides)

    # only pass kwargs the target constructor actually accepts
    sig = inspect.signature(cls.__init__)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        accepted = set(sig.parameters) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**kwargs)


def _register_builtin_datasets() -> None:
    """Import the dataset modules so their @register_dataset decorators run.

    Best-effort: a module that fails to import (e.g. missing optional dep) is
    skipped rather than breaking the whole factory.
    """
    import importlib
    for module in ("src.datasets.vision", "src.datasets.tabular", "src.datasets.detection"):
        try:
            importlib.import_module(module)
        except Exception:
            continue


_register_builtin_datasets()
