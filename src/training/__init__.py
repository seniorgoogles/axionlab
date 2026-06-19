"""Training module.

CANONICAL ENTRY POINT: ``src.core.Runner`` (train / validate / evaluate, phases,
callbacks, best.pth + last.pth).

Only the lightweight, current building blocks are imported eagerly
(TrainerConfig, CheckpointManager). The legacy ``Trainer`` / ``Model`` /
``Validator`` and the metric/callback helpers are loaded LAZILY (PEP 562) so that
importing the Runner does not pull the deprecated code (and therefore none of its
own heavy deps, e.g. the ``tqdm`` used by ``Validator``). They remain importable
for backward compatibility (and emit DeprecationWarnings).
"""

from .checkpoint_manager import CheckpointManager, CheckpointMetadata
from .trainer_config import TrainerConfig, create_default_config

# name -> submodule, imported on first access only
_LAZY = {
    "Trainer": ".trainer",
    "EarlyStoppingException": ".trainer",
    "Model": ".model",
    "TrainableModelConfig": ".model",
    "build_model": ".model",
    "Validator": ".validator",
    "Callback": ".callbacks",
    "MetricTracker": ".metric_tracker",
    "MetricHistory": ".metric_tracker",
}

__all__ = [
    "TrainerConfig",
    "create_default_config",
    "CheckpointManager",
    "CheckpointMetadata",
    *_LAZY.keys(),
]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
