"""
Unified Model class for training, evaluation, and deployment.

This module provides a unified Model class that encapsulates:
- Training functionality (via Trainer integration)
- Evaluation functionality (via Validator integration)
- Configuration management (save/load/build_from_config)
- Export capabilities

The Model class serves as the central interface for model lifecycle management,
replacing the need for separate Trainer and Validator classes in many use cases.

Components:
- Model: Main unified class for model lifecycle management
- TrainableTrainableModelConfig: Configuration for model training and evaluation
- build_model: Factory function for model creation from config
"""

from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer_config import TrainerConfig
from .trainer import Trainer
from .metric_tracker import MetricTracker
from .checkpoint_manager import CheckpointManager
from .validator import Validator

from src.utils.device_selector import DeviceSelector
from src.utils.timer import timer


@dataclass
class TrainableModelConfig(TrainerConfig):
    """Configuration for model training and evaluation.

    This configuration class extends TrainerConfig with model-specific settings
    and serves as the main interface for model configuration.

    Attributes:
        model_type: Type of model to create.
        batch_size: Batch size for training.
        metrics: List of evaluation metrics.
        num_workers: Number of data loading workers.
    """

    # Model-specific settings
    model_type: type = nn.Module
    batch_size: int = 32
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    num_workers: int = 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = super().to_dict()
        result["model_type"] = self.model_type.__name__
        result["batch_size"] = self.batch_size
        result["metrics"] = self.metrics
        result["num_workers"] = self.num_workers
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainableModelConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary configuration.

        Returns:
            TrainableModelConfig instance.
        """
        return cls(**config_dict)

    def build_from_config(self, **model_kwargs) -> "Model":
        """Build a Model instance from this configuration.

        Args:
            **model_kwargs: Additional arguments to pass to model constructor.

        Returns:
            Model instance with model built.
        """
        return build_model(self, **model_kwargs)

    def save_config(self, config_path: Optional[Path] = None) -> Path:
        """Save model configuration to file.

        Args:
            config_path: Path to save config. Defaults to save_dir/experiment_name/config.json.

        Returns:
            Path to saved config file.
        """
        if config_path is None:
            config_path = Path(self.save_dir) / self.experiment_name / "config.json"

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return config_path


class Model:
    """Unified model class for training, evaluation, and deployment.

    This class encapsulates model lifecycle management including:
    - Model initialization and configuration
    - Training with checkpointing
    - Evaluation with metrics
    - Configuration persistence
    - Export to various formats

    Attributes:
        model: PyTorch model instance.
        config: TrainableModelConfig instance.
        trainer: Trainer instance for training.
        validator: Validator instance for evaluation.
        checkpoint_manager: CheckpointManager instance.
        metric_tracker: MetricTracker instance.

    Example:
        >>> # Using with config
        >>> config = TrainableModelConfig(model_name="my_model", epochs=10)
        >>> model = Model(config)
        >>> model.build()
        >>> model.train(train_loader, val_loader, optimizer, criterion)
        >>> results = model.evaluate(val_loader)
        >>> model.save_checkpoint()
        >>> model.export("exported_model.pth")
    """

    def __init__(self, config: Union[TrainableModelConfig, Dict[str, Any]] = None):
        """Initialize the Model.

        Args:
            config: TrainableModelConfig instance or dictionary with configuration.
        """
        if config is None:
            config = TrainableModelConfig()
        elif isinstance(config, dict):
            config = TrainableModelConfig.from_dict(config)

        self.config = config
        self.model = None
        self.device = None

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize training and evaluation components."""
        # Initialize trainer
        self.trainer = Trainer(config=self.config)

        self.checkpoint_manager = CheckpointManager(
            self.config.save_dir, self.config.model_name, self.config
        )
        self.metric_tracker = MetricTracker(self.config.metrics)
        self.validator = Validator(
            device=self.config.device,
            metrics=self.config.metrics,
        )

    def build(self, **kwargs) -> "Model":
        """Build and initialize the model.

        Args:
            **kwargs: Additional arguments to pass to model constructor.

        Returns:
            Self for method chaining.
        """
        self.model = self.config.model_type(**kwargs)
        self.device = DeviceSelector.get_device()
        return self

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: Optional[int] = None,
        validate_every: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: PyTorch optimizer.
            criterion: Loss criterion.
            scheduler: Optional learning rate scheduler.
            num_epochs: Number of epochs to train.
            validate_every: How often to validate.

        Returns:
            Dictionary containing training results.
        """
        num_epochs = num_epochs or self.config.epochs
        validate_every = validate_every or self.config.validate_every

        if self.model is None:
            raise ValueError("Model must be built before training")

        # Use local trainer to train the model
        train_results = self.trainer.train(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=num_epochs,
            validate_every=validate_every,
        )

        return train_results

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
        num_batches: int = -1,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            dataloader: DataLoader for evaluation data.
            criterion: Optional loss criterion for loss computation.
            num_batches: Number of batches to evaluate.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model must be built before evaluation")

        return self.validator.validate(
            model=self.model,
            dataloader=dataloader,
            criterion=criterion,
            num_batches=num_batches,
            show_progress=show_progress,
        )

    def save_checkpoint(self, filename: Optional[str] = None) -> Path:
        """Save model checkpoint.

        Args:
            filename: Name of checkpoint file. Defaults to best.pth or last.pth.

        Returns:
            Path to saved checkpoint.
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        if filename is None:
            filename = "best.pth" if self.config.save_best else "last.pth"

        save_path = self.checkpoint_manager.save_checkpoint(
            self.model, filename=filename
        )
        return save_path

    def load_checkpoint(self, filename: str = "best.pth") -> None:
        """Load model from checkpoint.

        Args:
            filename: Name of checkpoint file to load.
        """
        if self.model is None:
            raise ValueError("Model must be built before loading")

        self.checkpoint_manager.load_model(self.model, filename)
        print(f"Loaded model from {filename}")

    def save_config(self, config_path: Optional[Path] = None) -> Path:
        """Save model configuration to file.

        Args:
            config_path: Path to save config. Defaults to save_dir/experiment_name/config.json.

        Returns:
            Path to saved config file.
        """
        return self.config.save_config(config_path)

    def save_metrics(self, metrics_path: Optional[Path] = None) -> Path:
        """Save training metrics to file.

        Args:
            metrics_path: Path to save metrics. Defaults to save_dir/metrics.json.

        Returns:
            Path to saved metrics file.
        """
        if metrics_path is None:
            metrics_path = Path(self.config.save_dir) / "metrics.json"

        save_dir = Path(metrics_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        self.metric_tracker.save(metrics_path)
        return Path(metrics_path)

    def export(
        self,
        export_path: Union[Path, str],
        export_type: str = "pt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Export model to file.

        Args:
            export_path: Path to export file.
            export_type: Type of export ("pt", "onnx", "pth").
            metadata: Optional metadata to save with model.

        Returns:
            Path to exported file.
        """
        if self.model is None:
            raise ValueError("Model must be built before exporting")

        export_path = Path(export_path)
        export_dir = export_path.parent
        export_dir.mkdir(parents=True, exist_ok=True)

        if export_type == "pt":
            # Save as state dict
            state_dict = self.model.state_dict()
            torch.save(state_dict, export_path)
        elif export_type == "pth":
            # Save as full model
            torch.save(self.model, export_path)
        elif export_type == "onnx":
            # Export as ONNX
            self._export_to_onnx(export_path)
        else:
            raise ValueError(f"Unsupported export type: {export_type}")

        # Save metadata if provided
        if metadata:
            metadata_path = export_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return export_path

    def _export_to_onnx(self, export_path: Path):
        """Export model to ONNX format.

        Args:
            export_path: Path to save ONNX model.
        """
        dummy_input = torch.randn(1, *self.model.input_shape if hasattr(self.model, "input_shape") else (1, 28, 28))
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )

    def get_device(self) -> torch.device:
        """Get the device used by the model.

        Returns:
            torch.device instance.
        """
        if self.device is None:
            self.device = DeviceSelector.get_device()
        return self.device

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None.
        """
        return self.trainer.get_best_checkpoint_path()

    def get_metrics_history(self) -> Dict[str, Any]:
        """Get training metrics history.

        Returns:
            Dictionary containing metrics history.
        """
        return self.trainer.get_metrics_history()

    def __repr__(self) -> str:
        return (
            f"Model(name={self.config.model_name}, "
            f"epochs={self.config.epochs}, "
            f"metrics={self.config.metrics})"
        )


def build_model(
    config: Union[TrainableModelConfig, Dict[str, Any]],
    **model_kwargs
) -> Model:
    """Factory function to create and build a Model.

    Args:
        config: TrainableModelConfig or dictionary with configuration.
        **model_kwargs: Additional arguments to pass to model constructor.

    Returns:
        Model instance with model built.
    """
    model = Model(config)
    model.build(**model_kwargs)
    return model