"""
Configuration class for trainer behavior.

This module provides a centralized configuration class for the Trainer,
allowing flexible customization of training parameters, checkpointing,
and metric tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any


@dataclass
class TrainerConfig:
    """Configuration for trainer behavior.

    Attributes:
        save_dir: Directory to save model checkpoints and experiments.
        experiment_name: Name for the current experiment.
        save_best: Whether to save the best model based on the best metric.
        save_last: Whether to save the last model at the end of training.
        best_metric_name: The metric to use for determining the best model.
        best_metric_mode: Whether to maximize or minimize the best metric.
        track_metrics: List of metrics to track during training.
        save_metrics_history: Whether to save metrics history to JSON.
        validate_every: How often to run validation (0 = never).
        early_stopping_patience: Number of epochs to wait before stopping.
        early_stopping_metric: Metric to use for early stopping.
        resume_from: Path to checkpoint to resume training from.
        device: Device to use for training (None = auto-detect).
        save_intermediate: Whether to save intermediate checkpoints.
        intermediate_every: How often to save intermediate checkpoints.
        model_name: Name of the model for checkpoint naming.
    """

    # Directory settings
    save_dir: str = "experiments"
    experiment_name: str = "default_experiment"
    model_name: str = "model"

    # Training settings
    epochs: int = 100
    lr: float = 1e-3

    # Model checkpointing
    save_best: bool = True
    save_last: bool = True
    best_metric_name: str = "accuracy"
    best_metric_mode: str = "max"  # "max" or "min"

    # Metric tracking
    track_metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    save_metrics_history: bool = True

    # Validation
    validate_every: int = 1
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "loss"

    # Other
    resume_from: Optional[str] = None
    device: Optional[str] = None
    save_intermediate: bool = False
    intermediate_every: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.best_metric_mode not in {"max", "min"}:
            raise ValueError("best_metric_mode must be 'max' or 'min'")

        if self.validate_every < 0:
            raise ValueError("validate_every must be non-negative")

        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

        if self.intermediate_every < 0:
            raise ValueError("intermediate_every must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "save_dir": self.save_dir,
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "epochs": self.epochs,
            "lr": self.lr,
            "save_best": self.save_best,
            "save_last": self.save_last,
            "best_metric_name": self.best_metric_name,
            "best_metric_mode": self.best_metric_mode,
            "track_metrics": self.track_metrics,
            "save_metrics_history": self.save_metrics_history,
            "validate_every": self.validate_every,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_metric": self.early_stopping_metric,
            "resume_from": self.resume_from,
            "device": self.device,
            "save_intermediate": self.save_intermediate,
            "intermediate_every": self.intermediate_every,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary configuration.

        Returns:
            TrainerConfig instance.
        """
        return cls(**config_dict)


def create_default_config() -> TrainerConfig:
    """Create a default configuration.

    Returns:
        Default TrainerConfig instance.
    """
    return TrainerConfig()