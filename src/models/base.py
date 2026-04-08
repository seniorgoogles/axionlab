"""Base model class for neural network architectures.

This module provides the abstract BaseModel class that all model implementations
should inherit from, along with shared functionality for model registration
and creation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import inspect
from pathlib import Path
import json
import torch
import torch.nn as nn


@dataclass
class ModelInfo:
    """Metadata about a model class.

    Attributes:
        name: Model name (from class name)
        description: Human-readable description
        input_shape: Expected input shape (batch, channels, height, width)
        output_shape: Expected output shape (batch, num_classes)
        num_params: Number of trainable parameters
        bitwidth: Default bitwidth for quantization
    """
    name: str
    description: str
    input_shape: Tuple[int, ...] = (1, 1, 28, 28)
    output_shape: Tuple[int, ...] = (1, 10)
    num_params: int = 0
    bitwidth: int = 32


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models.

    This class provides a common interface for all model types, including:
    - Standard PyTorch nn.Module interface
    - Abstract method for forward pass
    - Helper methods for model information and utilities

    Subclasses should implement the abstract methods and provide
    model-specific configuration.

    Example:
        >>> class MyModel(BaseModel):
        ...     def __init__(self, num_classes=10):
        ...         super().__init__()
        ...         self.conv1 = nn.Conv2d(1, 32, 3)
        ...         self.fc = nn.Linear(32 * 26 * 26, num_classes)
        ...
        ...     def forward(self, x):
        ...         x = self.conv1(x)
        ...         return torch.flatten(x, 1)
    """

    def __init__(self):
        """Initialize the base model."""
        super().__init__()
        self._info = self._get_model_info()

    @property
    def model_name(self) -> str:
        """Get the name of this model."""
        return self.__class__.__name__

    @property
    def model_info(self) -> ModelInfo:
        """Get model metadata."""
        return self._info

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def get_num_params(self) -> int:
        """Get number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_macs(self, input_shape: Optional[Tuple[int, ...]] = None) -> int:
        """Get number of multiply-accumulate operations (MACs).

        Args:
            input_shape: Input shape to compute MACs for. If None, uses model_info.

        Returns:
            Number of MACs
        """
        if input_shape is None:
            input_shape = self.model_info.input_shape

        # Simplified MAC computation
        macs = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                out_h = input_shape[2] - module.kernel_size[0] + 1
                out_w = input_shape[3] - module.kernel_size[1] + 1
                macs += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w
            elif isinstance(module, nn.Linear):
                macs += module.in_features * module.out_features

        return macs

    def get_memory_usage(self, input_shape: Optional[Tuple[int, ...]] = None, device: Optional[torch.device] = None) -> Dict[str, float]:
        """Get approximate memory usage.

        Args:
            input_shape: Input shape to compute memory for.
            device: Target device.

        Returns:
            Dictionary with memory usage in MB
        """
        if device is None:
            device = next(self.parameters()).device

        num_params = self.get_num_params()
        num_macs = self.get_num_macs(input_shape)

        # Parameter memory (float32 = 4 bytes)
        param_memory = num_params * 4 / (1024 ** 2)

        # Activation memory (rough estimate)
        if input_shape is None:
            input_shape = self.model_info.input_shape
        batch_size = input_shape[0]

        activation_memory = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                out_h = input_shape[2] - module.kernel_size[0] + 1
                out_w = input_shape[3] - module.kernel_size[1] + 1
                activation = batch_size * module.out_channels * out_h * out_w * 4 / (1024 ** 2)
            elif isinstance(module, nn.Linear):
                activation = batch_size * module.out_features * 4 / (1024 ** 2)
            activation_memory += activation

        total_memory = param_memory + activation_memory

        return {
            "parameters": param_memory,
            "activations": activation_memory,
            "total": total_memory,
            "input_size_mb": batch_size * input_shape[1] * input_shape[2] * input_shape[3] * 4 / (1024 ** 2)
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save model state to file.

        Args:
            path: Path to save model.

        Returns:
            Path to saved model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: Union[str, Path], strict: bool = True) -> "BaseModel":
        """Load model state from file.

        Args:
            path: Path to load model from.
            strict: Whether to strictly match state dict keys.

        Returns:
            Self for method chaining
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state_dict = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(state_dict, strict=strict)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert model state to dictionary.

        Returns:
            Dictionary of state dict
        """
        return self.state_dict()

    def from_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> "BaseModel":
        """Load model from state dictionary.

        Args:
            state_dict: State dictionary.
            strict: Whether to strictly match state dict keys.

        Returns:
            Self for method chaining
        """
        self.load_state_dict(state_dict, strict=strict)
        return self

    def _get_model_info(self) -> ModelInfo:
        """Get model metadata. Override to customize.

        Returns:
            ModelInfo instance
        """
        return ModelInfo(
            name=self.model_name,
            description=self.__doc__ or self.__class__.__name__
        )

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None, device: Optional[torch.device] = None) -> str:
        """Generate model summary.

        Args:
            input_shape: Input shape.
            device: Target device.

        Returns:
            Formatted summary string
        """
        if device is None:
            device = next(self.parameters()).device

        print(f"Model: {self.model_name}")
        print(f"Device: {device}")
        print(f"Input shape: {input_shape or self.model_info.input_shape}")
        print(f"Output shape: {self.model_info.output_shape}")
        print(f"Parameters: {self.get_num_params():,}")
        print(f"MACs: {self.get_num_macs(input_shape):,}")

        memory = self.get_memory_usage(input_shape, device)
        print(f"Memory usage: {memory['total']:.2f} MB")

        return str(self)

    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(input_shape={self.model_info.input_shape}, output_shape={self.model_info.output_shape})"


def _get_model_info_from_class(cls: type) -> ModelInfo:
    """Extract model info from class metadata.

    Args:
        cls: Model class.

    Returns:
        ModelInfo instance
    """
    doc = cls.__doc__ or ""

    # Try to extract from docstring
    name = cls.__name__
    description = doc.split("\n")[0] if doc else cls.__name__

    # Try to extract input/output shape from docstring
    input_shape = (1, 1, 28, 28)
    output_shape = (1, 10)

    if "Input shape:" in doc:
        import re
        input_match = re.search(r"Input shape:\s*\(([^)]+)\)", doc)
        if input_match:
            input_shape = tuple(map(int, input_match.group(1).split(",")))

    if "Output shape:" in doc:
        import re
        output_match = re.search(r"Output shape:\s*\(([^)]+)\)", doc)
        if output_match:
            output_shape = tuple(map(int, output_match.group(1).split(",")))

    return ModelInfo(name=name, description=description, input_shape=input_shape, output_shape=output_shape)


def get_module_name(cls: type) -> str:
    """Get module name for a model class.

    Args:
        cls: Model class.

    Returns:
        Module name
    """
    return cls.__module__