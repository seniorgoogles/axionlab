"""Configuration module for axionlab.

This module provides configuration reading, validation, and writing utilities
for YAML/file-based configuration.

Usage:
    >>> from src.config import ConfigReader, ConfigValidator, YAMLModelConfig
    >>> config_dict = ConfigReader.read_config('config.yaml')
    >>> model_config = ConfigValidator.validate_and_create(config_dict, YAMLModelConfig)

Note: For runtime training configuration, use:
    >>> from src.training import TrainerConfig
"""

from .reader import ConfigReader
from .validator import ConfigValidator
from .writer import ConfigWriter
from .types import (
    LayerDefinition,
    DatasetConfig,
    TrainingConfig,
    QuantizationSettings,
    YAMLModelConfig,
)

__all__ = [
    # Core utilities
    'ConfigReader',
    'ConfigValidator',
    'ConfigWriter',
    # Dataclass configurations for YAML parsing
    'LayerDefinition',
    'DatasetConfig',
    'TrainingConfig',
    'QuantizationSettings',
    'YAMLModelConfig',
]