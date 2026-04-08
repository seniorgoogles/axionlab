"""
Metric tracking class for recording and analyzing training metrics.

This module provides a MetricTracker class to track metric history during training,
save/load metrics to/from JSON, and retrieve best values and epochs for any metric.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class MetricHistory:
    """Record of metric values for a single metric."""

    metric_name: str
    values: List[float] = field(default_factory=list)
    best_value: Optional[float] = None
    best_epoch: Optional[int] = None
    mode: str = "max"  # "max" or "min"


class MetricTracker:
    """Track metrics history during training.

    This class maintains a history of metric values for each tracked metric,
    saves metrics to JSON files, and provides utilities for retrieving
    best values and analyzing metric trends.

    Attributes:
        tracked_metrics: List of metric names to track.
        history: Dictionary mapping metric names to their history.
        metadata: Additional metadata about training runs.

    Example:
        >>> tracker = MetricTracker(["loss", "accuracy"])
        >>> tracker.update(loss=0.5, accuracy=0.9)
        >>> tracker.update(loss=0.3, accuracy=0.95)
        >>> best_loss, best_loss_epoch = tracker.get_best("loss")
        >>> tracker.save("metrics.json")
    """

    def __init__(self, tracked_metrics: List[str]):
        """Initialize the MetricTracker.

        Args:
            tracked_metrics: List of metric names to track.
        """
        self.tracked_metrics = tracked_metrics
        self.history: Dict[str, MetricHistory] = {
            metric: MetricHistory(metric_name=metric) for metric in tracked_metrics
        }
        self.metadata: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def update(self, **metrics: float) -> None:
        """Update metric values with current epoch information.

        Args:
            **metrics: Keyword arguments of metric_name=value pairs.
                      If 'epoch' is not provided, it will be auto-assigned.

        Example:
            >>> tracker.update(loss=0.5, accuracy=0.9, epoch=1)
        """
        # Auto-add epoch if not provided
        if "epoch" not in metrics and self._get_current_epoch() is not None:
            metrics["epoch"] = self._get_current_epoch()

        for metric_name, value in metrics.items():
            if metric_name in self.tracked_metrics:
                history = self.history[metric_name]
                history.values.append(value)

                # Update best if mode is max and value > best, or mode is min and value < best
                if history.best_value is None:
                    history.best_value = value
                    history.best_epoch = metrics.get("epoch")
                elif history.mode == "max" and value > history.best_value:
                    history.best_value = value
                    history.best_epoch = metrics.get("epoch")
                elif history.mode == "min" and value < history.best_value:
                    history.best_value = value
                    history.best_epoch = metrics.get("epoch")

    def _get_current_epoch(self) -> Optional[int]:
        """Get the current epoch number if available.

        Returns:
            Current epoch number or None.
        """
        # Try to find epoch from the first metric's history
        for history in self.history.values():
            if history.values:
                last_entry = history.values[-1]
                if isinstance(last_entry, dict) and "epoch" in last_entry:
                    return last_entry["epoch"]
                elif isinstance(last_entry, (int, float)) and last_entry < 1000:
                    # If value is small, assume it's epoch
                    return int(last_entry)
        return None

    def get_history(self, metric_name: str) -> List[float]:
        """Get the history of values for a specific metric.

        Args:
            metric_name: Name of the metric to retrieve.

        Returns:
            List of metric values.

        Raises:
            KeyError: If the metric is not tracked.
        """
        if metric_name not in self.history:
            raise KeyError(f"Metric '{metric_name}' is not tracked")
        return self.history[metric_name].values

    def get_best(self, metric_name: str) -> Tuple[float, Optional[int]]:
        """Get the best value and epoch for a metric.

        Args:
            metric_name: Name of the metric to get best value for.

        Returns:
            Tuple of (best_value, best_epoch).

        Raises:
            KeyError: If the metric is not tracked.
        """
        if metric_name not in self.history:
            raise KeyError(f"Metric '{metric_name}' is not tracked")

        history = self.history[metric_name]
        return history.best_value, history.best_epoch

    def get_metric_info(self, metric_name: str) -> MetricHistory:
        """Get the full history object for a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            MetricHistory object.

        Raises:
            KeyError: If the metric is not tracked.
        """
        if metric_name not in self.history:
            raise KeyError(f"Metric '{metric_name}' is not tracked")
        return self.history[metric_name]

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of all tracked metrics.

        Returns:
            Dictionary mapping metric names to summary statistics.
        """
        summary = {}
        for metric_name, history in self.history.items():
            values = history.values
            if values:
                summary[metric_name] = {
                    "best_value": history.best_value,
                    "best_epoch": history.best_epoch,
                    "final_value": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        return summary

    def set_mode(self, metric_name: str, mode: Literal["max", "min"]) -> None:
        """Set the mode for a metric (maximize or minimize).

        Args:
            metric_name: Name of the metric.
            mode: "max" to track best maximum, "min" to track best minimum.

        Raises:
            KeyError: If the metric is not tracked.
        """
        if metric_name not in self.history:
            raise KeyError(f"Metric '{metric_name}' is not tracked")
        self.history[metric_name].mode = mode

    def set_metadata(self, key: str, value: Any) -> None:
        """Set additional metadata.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        return self.metadata.get(key, default)

    def start_timer(self) -> None:
        """Record the start time of training."""
        self.start_time = datetime.now()

    def end_timer(self) -> None:
        """Record the end time of training."""
        self.end_time = datetime.now()

    def get_duration(self) -> Optional[float]:
        """Get the duration of training in seconds.

        Returns:
            Duration in seconds or None if timer not started/ended.
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def save(self, path: str) -> None:
        """Save metrics history to a JSON file.

        Args:
            path: Path to save the metrics JSON file.

        Example:
            >>> tracker.save("training_metrics.json")
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.get_duration(),
            "history": {
                metric_name: {
                    "values": history.values,
                    "best_value": history.best_value,
                    "best_epoch": history.best_epoch,
                    "mode": history.mode,
                }
                for metric_name, history in self.history.items()
            },
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load metrics history from a JSON file.

        Args:
            path: Path to load the metrics JSON file from.

        Example:
            >>> tracker.load("training_metrics.json")
        """
        save_path = Path(path)
        if not save_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")

        with open(save_path, "r") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.start_time = datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None
        self.end_time = datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None

        for metric_name, metric_data in data.get("history", {}).items():
            if metric_name not in self.tracked_metrics:
                continue

            history = self.history[metric_name]
            history.values = metric_data.get("values", [])
            history.best_value = metric_data.get("best_value")
            history.best_epoch = metric_data.get("best_epoch")
            history.mode = metric_data.get("mode", "max")

    def reset(self) -> None:
        """Reset all tracked metrics and history."""
        for history in self.history.values():
            history.values = []
            history.best_value = None
            history.best_epoch = None
        self.metadata = {}
        self.start_time = None
        self.end_time = None

    def __len__(self) -> int:
        """Get the number of recorded entries for all metrics.

        Returns:
            Total number of recorded entries.
        """
        return sum(len(h.values) for h in self.history.values())