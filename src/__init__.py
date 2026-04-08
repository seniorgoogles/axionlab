"""Axionlab package.

Main entry point for importing configuration and model classes.

Usage:
    >>> from src.engine.config import ConfigReader, ConfigValidator, ModelConfig, TrainerConfig
    >>> from src.engine.trainer import Model
"""

from .engine.config import (
    ConfigReader,
    ConfigValidator,
    ConfigWriter,
    LayerDefinition,
    DatasetConfig,
    TrainingConfig,
    QuantizationConfig,
    ModelConfig,
    TrainerConfig,
    create_default_config,
)

__all__ = [
    # Config module exports
    'ConfigReader',
    'ConfigValidator',
    'ConfigWriter',
    'LayerDefinition',
    'DatasetConfig',
    'TrainingConfig',
    'QuantizationConfig',
    'ModelConfig',
    'TrainerConfig',
    'create_default_config',
]