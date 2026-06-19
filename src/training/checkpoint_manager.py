"""
Checkpoint manager for handling model checkpoint files.

This module provides a centralized CheckpointManager class for managing
model checkpoint files (.pth), including saving, loading, and discovering
available checkpoints.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch

from .trainer_config import TrainerConfig


@dataclass
class CheckpointMetadata:
    """Metadata stored with a checkpoint.

    Attributes:
        filename: Name of the checkpoint file.
        model_name: Name of the model.
        save_dir: Directory where checkpoint was saved.
        save_timestamp: When the checkpoint was created.
        epoch: Epoch number when checkpoint was created.
        metrics: Dictionary of metrics at checkpoint time.
        is_best: Whether this is a best checkpoint.
        is_last: Whether this is a last checkpoint.
        config: Serialized configuration used.
    """

    filename: str
    model_name: str
    save_dir: str
    save_timestamp: str
    epoch: int
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False
    is_last: bool = False
    config: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manage model checkpoint files.

    This class handles saving, loading, and discovering available checkpoints
    for model weights, along with storing metadata about each checkpoint.

    Attributes:
        save_dir: Directory where checkpoints are stored.
        model_name: Name of the model for checkpoint naming.
        checkpoints: List of available checkpoint filenames.

    Example:
        >>> manager = CheckpointManager("experiments/my_model", "my_model")
        >>> manager.save_checkpoint(model, "best.pth", epoch=10, accuracy=0.95)
        >>> manager.save_checkpoint(model, "last.pth", epoch=100, accuracy=0.90)
        >>> best = manager.get_best_checkpoint()
        >>> if manager.checkpoint_exists("best.pth"):
        ...     model.load_state_dict(manager.load_checkpoint("best.pth"))
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str,
        config: Optional[TrainerConfig] = None,
    ):
        """Initialize the CheckpointManager.

        Args:
            save_dir: Directory to save checkpoints.
            model_name: Name of the model for checkpoint naming.
            config: TrainerConfig instance for metadata.
        """
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.config = config

        self.checkpoints: List[str] = []
        self.metadata: Dict[str, CheckpointMetadata] = {}
        self._best_value: Optional[float] = None
        self._discover_checkpoints()

    def _discover_checkpoints(self) -> None:
        """Discover all checkpoint files and load their metadata.

        Scans the save directory for .pth files and loads their associated
        metadata files.
        """
        self.checkpoints = []
        self.metadata = {}

        # Scan for .pth files
        for checkpoint_file in self.save_dir.glob("*.pth"):
            self.checkpoints.append(checkpoint_file.name)

        # Scan for checkpoint metadata files
        for metadata_file in self.save_dir.glob("*.json"):
            if metadata_file.name == "config.json":
                continue  # Skip config.json

            try:
                with open(metadata_file, "r") as f:
                    metadata_data = json.load(f)

                filename = metadata_file.stem + ".pth"
                if filename in self.checkpoints:
                    self.metadata[filename] = CheckpointMetadata(
                        filename=filename,
                        model_name=metadata_data.get("model_name", self.model_name),
                        save_dir=str(self.save_dir),
                        save_timestamp=metadata_data.get("save_timestamp", ""),
                        epoch=metadata_data.get("epoch", 0),
                        metrics=metadata_data.get("metrics", {}),
                        is_best=metadata_data.get("is_best", False),
                        is_last=metadata_data.get("is_last", False),
                        config=metadata_data.get("config", {}),
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort checkpoints by name
        self.checkpoints.sort()

    def checkpoint_exists(self, filename: str) -> bool:
        """Check if a checkpoint file exists.

        Args:
            filename: Name of the checkpoint file to check.

        Returns:
            True if checkpoint exists, False otherwise.
        """
        return (self.save_dir / filename).exists()

    def save_checkpoint(
        self,
        model,
        filename: str,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_last: bool = False,
    ) -> None:
        """Save model checkpoint to file.

        Args:
            model: PyTorch model to save.
            filename: Name of the checkpoint file.
            epoch: Current epoch number.
            metrics: Dictionary of metrics at checkpoint time.
            is_best: Whether this is the best checkpoint.
            is_last: Whether this is the last checkpoint.

        Example:
            >>> manager.save_checkpoint(model, "best.pth", epoch=10, accuracy=0.95)
        """
        save_path = self.save_dir / filename
        torch.save(model.state_dict(), save_path)

        # Save metadata
        metadata = CheckpointMetadata(
            filename=filename,
            model_name=self.model_name,
            save_dir=str(self.save_dir),
            save_timestamp=datetime.now().isoformat(),
            epoch=epoch,
            metrics=metrics or {},
            is_best=is_best,
            is_last=is_last,
        )

        if self.config:
            metadata.config = self.config.to_dict()

        metadata_path = self.save_dir / f"{filename}.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Update checkpoint list
        if filename not in self.checkpoints:
            self.checkpoints.append(filename)
        self.metadata[filename] = metadata

    def save_best(self, model, metric_value: float, metric_name: str,
                  mode: str = "max", epoch: int = 0) -> bool:
        """Save model as best.pth if the metric improved over the previous best.

        Args:
            model: PyTorch model to save.
            metric_value: Current value of the metric.
            metric_name: Name of the metric.
            mode: "max" or "min" indicating how to determine improvement.
            epoch: Current epoch number.

        Returns:
            True if a new best checkpoint was written, False otherwise.
        """
        improved = (
            self._best_value is None
            or (mode == "max" and metric_value > self._best_value)
            or (mode == "min" and metric_value < self._best_value)
        )
        if not improved:
            return False

        self._best_value = metric_value
        # same filename -> save_checkpoint overwrites in place (no manual unlink,
        # which previously deleted the file we had just written).
        self.save_checkpoint(
            model, "best.pth", epoch=epoch,
            metrics={metric_name: metric_value}, is_best=True, is_last=False,
        )
        return True

    def reset_best(self) -> None:
        """Forget the tracked best value. Call when the model architecture changes
        (e.g. after quantizing in a new phase) so best.pth restarts for the new
        architecture instead of comparing across incompatible checkpoints."""
        self._best_value = None

    def save_last(self, model, epoch: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save model as last checkpoint.

        Args:
            model: PyTorch model to save.
            epoch: Current epoch number.
            metrics: Dictionary of metrics at checkpoint time.

        Example:
            >>> manager.save_last(model, epoch=100, accuracy=0.90)
        """
        # same filename -> save_checkpoint overwrites in place.
        self.save_checkpoint(
            model, "last.pth", epoch=epoch,
            metrics=metrics, is_best=False, is_last=True,
        )

    def save_intermediate(self, model, epoch: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """Save intermediate checkpoint.

        Args:
            model: PyTorch model to save.
            epoch: Current epoch number.
            metrics: Dictionary of metrics at checkpoint time.
        """
        filename = f"epoch_{epoch:03d}.pth"
        self.save_checkpoint(
            model,
            filename,
            epoch=epoch,
            metrics=metrics,
            is_best=False,
            is_last=False,
        )

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """Load checkpoint metadata.

        Args:
            filename: Name of the checkpoint file to load metadata from.

        Returns:
            Dictionary of checkpoint metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        metadata_path = self.save_dir / f"{filename}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def load_model(self, model, filename: str) -> Dict[str, Any]:
        """Load model state dictionary from checkpoint.

        Args:
            model: PyTorch model to load weights into.
            filename: Name of the checkpoint file to load.

        Returns:
            Dictionary of checkpoint metadata.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        checkpoint_path = self.save_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(state_dict)

        metadata = self.load_checkpoint(filename)
        return metadata

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get information about the best checkpoint.

        Returns:
            Dictionary with 'filename', 'value', and 'epoch' keys,
            or None if no best checkpoint exists.
        """
        if "best.pth" in self.checkpoints:
            return {
                "filename": "best.pth",
                "value": self.metadata["best.pth"].metrics.get(
                    self.metadata["best.pth"].config.get("best_metric_name", "accuracy"),
                    0.0,
                ),
                "epoch": self.metadata["best.pth"].epoch,
            }
        return None

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get information about the last checkpoint.

        Returns:
            Dictionary with 'filename', 'value', and 'epoch' keys,
            or None if no last checkpoint exists.
        """
        if "last.pth" in self.checkpoints:
            return {
                "filename": "last.pth",
                "value": self.metadata["last.pth"].metrics.get(
                    self.metadata["last.pth"].config.get("best_metric_name", "accuracy"),
                    0.0,
                ),
                "epoch": self.metadata["last.pth"].epoch,
            }
        return None

    def get_available_checkpoints(self) -> List[str]:
        """Get list of available checkpoint filenames.

        Returns:
            List of checkpoint filenames.
        """
        return self.checkpoints.copy()

    def get_checkpoints_by_type(self, checkpoint_type: str) -> List[str]:
        """Get checkpoints of a specific type.

        Args:
            checkpoint_type: "best", "last", or "intermediate".

        Returns:
            List of checkpoint filenames of the specified type.
        """
        if checkpoint_type == "best":
            return ["best.pth"] if "best.pth" in self.checkpoints else []
        elif checkpoint_type == "last":
            return ["last.pth"] if "last.pth" in self.checkpoints else []
        elif checkpoint_type == "intermediate":
            return [cp for cp in self.checkpoints if cp.startswith("epoch_")]
        return []

    def get_checkpoint_metadata(self, filename: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint.

        Args:
            filename: Name of the checkpoint file.

        Returns:
            CheckpointMetadata object or None.
        """
        return self.metadata.get(filename)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Get detailed information about all checkpoints.

        Returns:
            List of dictionaries containing checkpoint details.
        """
        checkpoints_info = []
        for filename in self.checkpoints:
            metadata = self.metadata.get(filename)
            if metadata:
                checkpoints_info.append(asdict(metadata))
        return checkpoints_info

    def remove_checkpoint(self, filename: str) -> bool:
        """Remove a checkpoint file and its metadata.

        Args:
            filename: Name of the checkpoint file to remove.

        Returns:
            True if checkpoint was removed, False otherwise.
        """
        checkpoint_path = self.save_dir / filename
        metadata_path = self.save_dir / f"{filename}.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

        if filename in self.checkpoints:
            self.checkpoints.remove(filename)

        if filename in self.metadata:
            del self.metadata[filename]

        return True

    def clean_old_checkpoints(self, keep_best: int = 1, keep_last: int = 1) -> None:
        """Remove old checkpoints, keeping only the best and last.

        Args:
            keep_best: Number of best checkpoints to keep.
            keep_last: Number of last checkpoints to keep.
        """
        # Remove old best checkpoints
        best_checkpoints = self.get_checkpoints_by_type("best")
        for checkpoint in best_checkpoints[keep_best:]:
            self.remove_checkpoint(checkpoint)

        # Remove old last checkpoints
        last_checkpoints = self.get_checkpoints_by_type("last")
        for checkpoint in last_checkpoints[keep_last:]:
            self.remove_checkpoint(checkpoint)

        # Remove old intermediate checkpoints
        intermediate_checkpoints = self.get_checkpoints_by_type("intermediate")
        for checkpoint in intermediate_checkpoints:
            self.remove_checkpoint(checkpoint)

    def get_save_dir(self) -> Path:
        """Get the save directory.

        Returns:
            Path object representing the save directory.
        """
        return self.save_dir

    def get_num_checkpoints(self) -> int:
        """Get the number of checkpoints.

        Returns:
            Number of checkpoint files.
        """
        return len(self.checkpoints)

    def is_directory_empty(self) -> bool:
        """Check if the save directory is empty.

        Returns:
            True if no checkpoints exist, False otherwise.
        """
        return len(self.checkpoints) == 0