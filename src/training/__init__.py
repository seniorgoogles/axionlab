"""Training module.

This module provides a unified training infrastructure with checkpointing
and metrics tracking.

Components:
- Model: Unified model class for training, evaluation, and deployment
- TrainableModelConfig: Configuration dataclass for model training and evaluation
- Trainer: Main training loop with checkpointing (best.pth, last.pth)
- TrainerConfig: Configuration dataclass for trainer settings
- CheckpointManager: Handles saving/loading of model checkpoints
- MetricTracker: Tracks and records training metrics
- Validator: Model evaluation utilities
- Callback: Base class for training callbacks
"""

# Unified model classes
from .model import Model, TrainableModelConfig, build_model

# Trainer components
from .trainer_config import TrainerConfig, create_default_config
from .metric_tracker import MetricTracker, MetricHistory
from .checkpoint_manager import CheckpointManager, CheckpointMetadata
from .validator import Validator
from .trainer import Trainer, EarlyStoppingException
from .callbacks import Callback

__all__ = [
    # Unified model classes
    "Model",
    "TrainableModelConfig",
    "build_model",
    # Trainer components
    "TrainerConfig",
    "create_default_config",
    "MetricTracker",
    "MetricHistory",
    "CheckpointManager",
    "CheckpointMetadata",
    "Validator",
    "Trainer",
    "EarlyStoppingException",
    "Callback",
]