"""
Unified training loop with checkpointing and metrics tracking.

This module provides a comprehensive Trainer class that handles the entire training process,
including epoch loops, validation, checkpointing, and metrics tracking.
"""

from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer_config import TrainerConfig
from .metric_tracker import MetricTracker
from .checkpoint_manager import CheckpointManager
from .validator import Validator


class Trainer:
    """Unified training loop with checkpointing and metrics tracking.

    This class provides a comprehensive training loop that handles:
    - Training epochs
    - Validation
    - Checkpoint saving (best, last, intermediate)
    - Metrics tracking and history
    - Resume from checkpoint
    - Early stopping

    Attributes:
        config: TrainerConfig instance.
        checkpoint_manager: CheckpointManager instance.
        metric_tracker: MetricTracker instance.
        validator: Validator instance.

    Example:
        >>> config = TrainerConfig(save_dir="experiments", experiment_name="my_model")
        >>> trainer = Trainer(config)
        >>> trainer.train(model, train_loader, val_loader, optimizer, criterion)
    """

    def __init__(self, config: TrainerConfig):
        """Initialize the Trainer.

        .. deprecated::
            Use ``src.core.Runner`` instead. Runner is the single training entry
            point (train/validate/evaluate, phases, callbacks, best/last
            checkpoints). Trainer is kept only for backward compatibility.

        Args:
            config: TrainerConfig instance with training configuration.
        """
        import warnings
        warnings.warn(
            "training.Trainer is deprecated; use src.core.Runner instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.config = config
        self.checkpoint_manager = CheckpointManager(
            config.save_dir, config.model_name, config
        )
        self.metric_tracker = MetricTracker(config.track_metrics)
        self.validator = Validator(
            device=config.device,
            metrics=config.track_metrics,
        )

        self.start_time = None
        self.best_metric_value = None
        self.best_epoch = 0
        self.no_improvement_count = 0

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: Optional[int] = None,
        validate_every: Optional[int] = None,
        on_epoch_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train the model for specified epochs.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: PyTorch optimizer.
            criterion: Loss criterion.
            scheduler: Optional learning rate scheduler.
            num_epochs: Number of epochs to train (None = use config.epochs).
            validate_every: How often to validate (None = use config.validate_every).
            on_epoch_end: Optional callback function called after each epoch.

        Returns:
            Dictionary containing final metrics and training info.
        """
        num_epochs = num_epochs or self.config.epochs
        validate_every = validate_every or self.config.validate_every

        # Resume from checkpoint if configured
        if self.config.resume_from:
            self._resume_training(model, optimizer, criterion)

        # Initialize training state
        self.start_time = datetime.now()
        self.best_metric_value = None
        self.best_epoch = 0
        self.no_improvement_count = 0

        train_metrics = {}

        for epoch in range(self.current_epoch, num_epochs):
            epoch_start = datetime.now()

            # Train for one epoch
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion
            )

            # Validate
            val_metrics = {}
            if validate_every == 0 or epoch % validate_every == 0:
                val_metrics = self._validate_epoch(
                    model, val_loader, criterion
                )

            # Combine metrics for tracking
            all_metrics = {**train_metrics, **val_metrics}

            # Update tracker
            self.metric_tracker.update(**all_metrics, epoch=epoch)

            # Save checkpoints
            if self.config.save_best and val_metrics:
                self._save_best_checkpoint(
                    model, val_metrics, epoch
                )
            if self.config.save_last:
                self._save_last_checkpoint(model, epoch, all_metrics)

            # Save intermediate checkpoint
            if self.config.save_intermediate and epoch % self.config.intermediate_every == 0:
                self.checkpoint_manager.save_intermediate(model, epoch, all_metrics)

            # Early stopping check
            if self.config.early_stopping_patience and val_metrics:
                self._check_early_stopping(val_metrics)

            # Callback
            if on_epoch_end:
                on_epoch_end(epoch, all_metrics)

            # Scheduler step
            if scheduler:
                scheduler.step()

            epoch_duration = (datetime.now() - epoch_start).total_seconds()
            print(f"Epoch {epoch + 1}/{num_epochs} - Duration: {epoch_duration:.2f}s")

        self._finalize_training(train_metrics, val_metrics)

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric_value,
            "final_train_metrics": train_metrics,
            "final_val_metrics": val_metrics,
        }

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Train model for one epoch.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            optimizer: PyTorch optimizer.
            criterion: Loss criterion.

        Returns:
            Dictionary of training metrics.
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy}

    def _validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            model: PyTorch model to validate.
            val_loader: DataLoader for validation data.
            criterion: Loss criterion.

        Returns:
            Dictionary of validation metrics.
        """
        return self.validator.validate(model, val_loader, criterion)

    def _save_best_checkpoint(
        self,
        model: nn.Module,
        val_metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """Save model as best checkpoint if improved.

        Args:
            model: PyTorch model to save.
            val_metrics: Validation metrics.
            epoch: Current epoch number.
        """
        metric_name = self.config.best_metric_name
        mode = self.config.best_metric_mode

        if metric_name not in val_metrics:
            return

        self.checkpoint_manager.save_best(
            model,
            val_metrics[metric_name],
            metric_name,
            mode,
        )
        print(
            f"Saved best checkpoint at epoch {epoch}: "
            f"{metric_name} = {val_metrics[metric_name]:.4f}"
        )

    def _save_last_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save model as last checkpoint.

        Args:
            model: PyTorch model to save.
            epoch: Current epoch number.
            metrics: Combined metrics.
        """
        self.checkpoint_manager.save_last(
            model, epoch, {k: v for k, v in metrics.items() if isinstance(v, float)}
        )
        print(f"Saved last checkpoint at epoch {epoch}")

    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> None:
        """Check if training should stop early.

        Args:
            val_metrics: Validation metrics.
        """
        metric_name = self.config.early_stopping_metric
        patience = self.config.early_stopping_patience

        if metric_name not in val_metrics:
            return

        current_value = val_metrics[metric_name]
        mode = self.config.best_metric_mode

        improved = False
        if self.best_metric_value is None:
            improved = True
        elif mode == "max" and current_value > self.best_metric_value:
            improved = True
        elif mode == "min" and current_value < self.best_metric_value:
            improved = True

        if improved:
            self.best_metric_value = current_value
            self.best_epoch = self.current_epoch
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            print(
                f"No improvement for {self.no_improvement_count}/{patience} epochs"
            )
            if self.no_improvement_count >= patience:
                print(f"Early stopping triggered after {self.best_epoch} epochs")
                raise EarlyStoppingException(
                    f"Early stopping triggered after {self.best_epoch} epochs"
                )

    def _resume_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> None:
        """Resume training from checkpoint.

        Args:
            model: PyTorch model to load state into.
            optimizer: PyTorch optimizer to load state into.
            criterion: Loss criterion.
        """
        from ray.train import Checkpoint, get_checkpoint

        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                import pickle

                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)

                self.current_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["net_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
                print(f"Resumed training from epoch {self.current_epoch}")
        elif self.config.resume_from:
            # Legacy resume from file
            metadata = self.checkpoint_manager.load_checkpoint(self.config.resume_from)
            if metadata:
                self.current_epoch = metadata.get("epoch", 0) + 1
                model.load_state_dict(
                    torch.load(
                        self.checkpoint_manager.get_save_dir() / self.config.resume_from
                    )
                )
                print(f"Resumed training from epoch {self.current_epoch}")
        else:
            self.current_epoch = 0

    def _finalize_training(self, train_metrics, val_metrics) -> None:
        """Finalize training and save metrics.

        Args:
            train_metrics: Final training metrics.
            val_metrics: Final validation metrics.
        """
        self.metric_tracker.end_timer()
        self.metric_tracker.set_metadata(
            "total_epochs", self.config.epochs
        )
        self.metric_tracker.set_metadata(
            "start_time", self.start_time.isoformat()
        )
        self.metric_tracker.set_metadata(
            "end_time", datetime.now().isoformat()
        )
        self.metric_tracker.set_metadata(
            "duration_seconds", self.metric_tracker.get_duration()
        )

        # Save metrics history
        if self.config.save_metrics_history:
            metrics_path = (
                self.checkpoint_manager.get_save_dir() / "metrics.json"
            )
            self.metric_tracker.save(metrics_path)
            print(f"Saved metrics history to {metrics_path}")

        # Save config
        config_path = (
            self.checkpoint_manager.get_save_dir() / "config.json"
        )
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
            print(f"Saved config to {config_path}")

    @property
    def current_epoch(self) -> int:
        """Get the current epoch number.

        Returns:
            Current epoch number.
        """
        if not hasattr(self, "_current_epoch"):
            self._current_epoch = 0
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        """Set the current epoch number.

        Args:
            value: Epoch number to set.
        """
        self._current_epoch = value

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the best checkpoint.

        Returns:
            Path to best checkpoint or None.
        """
        best_path = self.checkpoint_manager.get_save_dir() / "best.pth"
        return best_path if best_path.exists() else None

    def get_metrics_history(self) -> Dict[str, Any]:
        """Get the metrics history.

        Returns:
            Metrics history dictionary.
        """
        return self.metric_tracker.get_summary()

    def save_model(self, model: nn.Module, filename: str = "model.pth") -> None:
        """Save model state dictionary.

        Args:
            model: PyTorch model to save.
            filename: Name of the file to save.
        """
        path = self.checkpoint_manager.get_save_dir() / filename
        torch.save(model.state_dict(), path)
        print(f"Saved model to {path}")

    def load_model(self, model: nn.Module, filename: str = "best.pth") -> None:
        """Load model state dictionary.

        Args:
            model: PyTorch model to load weights into.
            filename: Name of the checkpoint file to load.
        """
        self.checkpoint_manager.load_model(model, filename)
        print(f"Loaded model from {filename}")

    def __repr__(self) -> str:
        return (
            f"Trainer(config={self.config}, "
            f"checkpoints={self.checkpoint_manager.get_num_checkpoints()})"
        )


class EarlyStoppingException(Exception):
    """Exception raised when early stopping is triggered."""

    pass


# Import json for config saving
import json