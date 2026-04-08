"""Engine module for training and validation.

This module provides utilities for model training, validation, and checkpoint management.

Components:
- Model: Unified model class for training, evaluation, and deployment
- Trainer: Training loop with checkpointing (saves best.pth and last.pth after each epoch)
- CheckpointManager: Handles model checkpoint files
- MetricTracker: Tracks training metrics history
- Validator: Model evaluation utilities
- Callback: Base class for training callbacks

Note: TrainerConfig and ModelConfig have been moved to src.config.types for better organization.
"""

# Import from trainer module for backward compatibility
from .trainer import (
    Model,
    build_model,
    Trainer,
    EarlyStoppingException,
    MetricTracker,
    MetricHistory,
    CheckpointManager,
    CheckpointMetadata,
    Validator,
    Callback,
)

# Import configs from config module for backward compatibility
from src.engine.config import ModelConfig, TrainerConfig, create_default_config

__all__ = [
    # From trainer
    "Model",
    "build_model",
    "Trainer",
    "EarlyStoppingException",
    "MetricTracker",
    "MetricHistory",
    "CheckpointManager",
    "CheckpointMetadata",
    "Validator",
    "Callback",
    # From config (backward compatibility)
    "ModelConfig",
    "TrainerConfig",
    "create_default_config",
]