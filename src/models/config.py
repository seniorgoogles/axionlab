"""Model configuration classes.

This module provides configuration classes for model creation and training,
including default settings and serialization methods.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, Union, List
from enum import Enum

from src.models.base import BaseModel


class ModelTrainingMode(Enum):
    """Training mode options."""
    TRAIN = "train"
    EVAL = "eval"
    INFERENCE = "inference"


@dataclass
class ModelConfig:
    """Configuration for model creation and training.

    This class provides a flexible configuration interface for model creation,
    training parameters, and common settings.

    Attributes:
        model_type: Type of model class to instantiate.
        model_name: Name for logging and checkpointing.
        num_classes: Number of output classes.
        input_shape: Input shape for the model.
        device: Device to run the model on.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        save_dir: Directory to save checkpoints.
        save_best: Whether to save best model.
        save_interval: Save interval in epochs.
        validate_every: Validation frequency.
        dropout_rate: Dropout rate for regularization.
        optimizer: Optimizer type (e.g., 'adam', 'sgd').
        metrics: List of metrics to track.
        quantization: Quantization configuration.
        training_mode: Training mode.
    """
    # Model specification
    model_type: Type[BaseModel] = BaseModel
    model_name: str = "model"
    num_classes: int = 10
    input_shape: tuple = (1, 1, 28, 28)
    device: Optional[str] = None

    # Training parameters
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_interval: int = 1
    validate_every: int = 1
    dropout_rate: float = 0.5

    # Optimization
    optimizer: str = "adam"

    # Metrics and evaluation
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])

    # Quantization
    quantization: Optional[Dict[str, Any]] = None

    # Training mode
    training_mode: ModelTrainingMode = ModelTrainingMode.TRAIN

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not issubclass(self.model_type, BaseModel):
            raise TypeError(f"model_type must be a subclass of BaseModel, got {self.model_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {
            "model_type": self.model_type.__name__,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "device": self.device,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "save_dir": self.save_dir,
            "save_best": self.save_best,
            "save_interval": self.save_interval,
            "validate_every": self.validate_every,
            "dropout_rate": self.dropout_rate,
            "optimizer": self.optimizer,
            "metrics": self.metrics,
            "quantization": self.quantization,
            "training_mode": self.training_mode.value,
        }
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary configuration.

        Returns:
            ModelConfig instance.
        """
        # Convert string model_type back to class
        if "model_type" in config_dict:
            model_type_name = config_dict["model_type"]
            try:
                from src.models.registry import ModelRegistry
                config_dict["model_type"] = ModelRegistry.get_model_class(model_type_name)
            except (ImportError, AttributeError):
                pass

        # Convert training_mode string back to enum
        if "training_mode" in config_dict:
            training_mode = config_dict["training_mode"]
            if isinstance(training_mode, str):
                try:
                    config_dict["training_mode"] = ModelTrainingMode(training_mode.upper())
                except ValueError:
                    pass

        return cls(**config_dict)

    def build_model(self, **kwargs) -> BaseModel:
        """Build a model instance from this configuration.

        Args:
            **kwargs: Additional arguments to pass to model constructor.

        Returns:
            Model instance with model built.
        """
        return ModelConfig.build_model(self, **kwargs)

    def save_config(self, config_path: Optional[str] = None) -> str:
        """Save model configuration to file.

        Args:
            config_path: Path to save config. Defaults to save_dir/config.json.

        Returns:
            Path to saved config file.
        """
        if config_path is None:
            from pathlib import Path
            config_path = Path(self.save_dir) / f"{self.model_name}_config.json"

        import json
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return config_path

    @classmethod
    def create(cls, model_name: str = "model", **kwargs) -> "ModelConfig":
        """Create a ModelConfig with default settings.

        Args:
            model_name: Name for the model.
            **kwargs: Additional configuration parameters.

        Returns:
            ModelConfig instance.
        """
        return cls(model_name=model_name, **kwargs)


class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        mode: Quantization mode (e.g., 'dynamic', 'static', 'learned').
        bitwidth: Target bitwidth (e.g., 8, 4).
        weight_quant: Weight quantization config.
        activation_quant: Activation quantization config.
    """

    def __init__(
        self,
        mode: str = "dynamic",
        bitwidth: int = 8,
        weight_quant: Optional[Dict[str, Any]] = None,
        activation_quant: Optional[Dict[str, Any]] = None
    ):
        self.mode = mode
        self.bitwidth = bitwidth
        self.weight_quant = weight_quant or {}
        self.activation_quant = activation_quant or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert quantization config to dictionary."""
        return {
            "mode": self.mode,
            "bitwidth": self.bitwidth,
            "weight_quant": self.weight_quant,
            "activation_quant": self.activation_quant,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QuantizationConfig":
        """Create QuantizationConfig from dictionary."""
        return cls(**config_dict)


# Predefined configurations for common models
class ModelConfigurations:
    """Predefined configurations for common models."""

    @staticmethod
    def lenet5(num_classes: int = 10) -> ModelConfig:
        """Get configuration for LeNet-5."""
        return ModelConfig(
            model_name="lenet5",
            num_classes=num_classes,
            input_shape=(1, 1, 28, 28),
            learning_rate=0.001,
            epochs=20,
        )

    @staticmethod
    def resnet18(num_classes: int = 10) -> ModelConfig:
        """Get configuration for ResNet-18."""
        return ModelConfig(
            model_name="resnet18",
            num_classes=num_classes,
            input_shape=(3, 224, 224),
            learning_rate=0.001,
            epochs=50,
            batch_size=64,
        )

    @staticmethod
    def mobilenet_v3(num_classes: int = 1000) -> ModelConfig:
        """Get configuration for MobileNetV3."""
        return ModelConfig(
            model_name="mobilenet_v3",
            num_classes=num_classes,
            input_shape=(3, 224, 224),
            learning_rate=0.001,
            epochs=30,
            batch_size=32,
        )