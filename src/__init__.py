"""Axionlab package.

Usage:
    >>> from src.core import Runner, Phase          # train / validate / evaluate
    >>> from src.core.build import build_from_config, build_graph
    >>> from src.datasets import build_dataset
    >>> from src.config import ConfigReader
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