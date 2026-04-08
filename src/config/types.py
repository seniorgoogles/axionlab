"""Configuration types for axionlab.

This module defines dataclass configuration types for YAML/file-based configuration:
- LayerDefinition: Definition for a single layer in the model
- DatasetConfig: Dataset configuration
- TrainingConfig: Training hyperparameters configuration
- QuantizationSettings: Quantization settings from config files
- YAMLModelConfig: Complete model configuration from YAML files

Note: For runtime training configuration, use TrainerConfig from src.training.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LayerDefinition:
    """Definition for a single layer in the model.

    Attributes:
        from_layer: Input source (-1 for previous layer)
        repeats: Number of repetitions
        module_type: PyTorch module type (Linear, Conv2d, etc.)
        name: Layer name
        args: Layer arguments
    """
    from_layer: int
    repeats: int
    module_type: str
    name: str
    args: Dict[str, Any]


@dataclass
class DatasetConfig:
    """Dataset configuration.

    Attributes:
        name: Dataset name
        train_path: Path to training data
        test_path: Path to test data
        root_path: Root path for dataset
        num_workers: Number of worker processes for data loading
        batch_size: Batch size for training and evaluation
        distributed: Whether to use distributed training
    """
    name: str
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    root_path: Optional[str] = None
    num_workers: int = 4
    batch_size: List[int] = field(default_factory=lambda: [32, 32])
    distributed: bool = False


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        weight_decay: Weight decay regularization
    """
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    weight_decay: float = 0.0


@dataclass
class QuantizationSettings:
    """Quantization settings from config files.

    Attributes:
        enabled: Whether quantization is enabled
        weight_bit_width: Bit width for weights
        act_bit_width: Bit width for activations
        bias_bit_width: Bit width for bias
    """
    enabled: bool = False
    weight_bit_width: int = 8
    act_bit_width: int = 8
    bias_bit_width: int = 32


@dataclass
class YAMLModelConfig:
    """Complete model configuration from YAML files.

    This is used for parsing model architecture from configuration files.
    For runtime model configuration, use ModelConfig from src.models.config.

    Attributes:
        name: Model name
        dataset: Dataset configuration
        training: Training configuration
        num_classes: Number of output classes
        backbone: List of layer definitions for backbone
        neck: Optional list of layer definitions for neck
        head: Optional list of layer definitions for head
        quantization: Optional quantization settings
        weights_path: Path to pretrained weights
    """
    name: str
    dataset: DatasetConfig
    training: TrainingConfig
    num_classes: int
    backbone: List[LayerDefinition]
    neck: Optional[List[LayerDefinition]] = None
    head: Optional[List[LayerDefinition]] = None
    quantization: Optional[QuantizationSettings] = None
    weights_path: Optional[str] = None

    @property
    def all_layers(self) -> List[LayerDefinition]:
        """Get all layers in order: backbone + neck + head."""
        layers = list(self.backbone)
        if self.neck:
            layers.extend(self.neck)
        if self.head:
            layers.extend(self.head)
        return layers


# Note: TrainerConfig has been moved to src.training.trainer_config
# Import it from there: from src.training import TrainerConfig