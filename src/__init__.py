"""Axionlab package.

Main entry point for importing configuration and model classes.

Usage:
    >>> from src.config import ConfigReader, ConfigValidator, YAMLModelConfig
    >>> from src.training import Trainer, TrainerConfig
    >>> from src.models import ModelConfig
"""

from .config import (
    ConfigReader,
    ConfigValidator,
    ConfigWriter,
    LayerDefinition,
    DatasetConfig,
    TrainingConfig,
    QuantizationSettings,
    YAMLModelConfig,
)

__all__ = [
    # Config module exports (YAML parsing)
    'ConfigReader',
    'ConfigValidator',
    'ConfigWriter',
    'LayerDefinition',
    'DatasetConfig',
    'TrainingConfig',
    'QuantizationSettings',
    'YAMLModelConfig',
]