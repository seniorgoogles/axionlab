"""
Validator class for model evaluation on a dataset.

This module provides a Validator class for evaluating models on validation/test datasets,
with support for configurable metrics and checkpoint checking.
"""

from typing import Dict, List, Optional, Callable, Any

import torch
from tqdm import tqdm

from src.utils.device_selector import DeviceSelector


class Validator:
    """Validate model on a dataset.

    This class provides methods for evaluating models on validation/test datasets,
    with configurable metrics and the ability to check for available checkpoints
    before validation.

    Attributes:
        device: Device to use for validation.
        metrics: List of metrics to compute.

    Example:
        >>> validator = Validator()
        >>> accuracy, loss = validator.validate(model, val_loader)
        >>> # Check for available checkpoints
        >>> has_weights = validator.check_weights_available(checkpoint_path)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        metric_fn: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize the Validator.

        Args:
            device: Device to use for validation (None = auto-detect).
            metrics: List of metric names to compute.
            metric_fn: Dictionary mapping metric names to callable functions.
                      If provided, these functions will be used instead of default implementations.
                      Default functions: "accuracy", "loss", "f1_score".
        """
        self.device = (
            DeviceSelector.get_device() if device is None else device
        )
        self.metrics = metrics or ["accuracy", "loss"]
        self.metric_fn = metric_fn or _default_metric_functions()


def _default_metric_functions() -> Dict[str, Callable]:
    """Get default metric computation functions.

    Returns:
        Dictionary mapping metric names to functions.
    """
    return {
        "accuracy": _accuracy,
        "loss": _loss,
    }


def _accuracy(outputs, targets) -> float:
    """Compute accuracy metric.

    Args:
        outputs: Model outputs.
        targets: Ground truth labels.

    Returns:
        Accuracy value.
    """
    _, predicted = outputs.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def _loss(outputs, targets, criterion) -> float:
    """Compute loss metric.

    Args:
        outputs: Model outputs.
        targets: Ground truth labels.
        criterion: Loss criterion.

    Returns:
        Loss value.
    """
    loss = criterion(outputs, targets)
    return loss.item()


def check_weights_available(checkpoint_path: str) -> bool:
    """Check if weights exist at a checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        True if checkpoint file exists and is not empty, False otherwise.
    """
    import os

    if not os.path.exists(checkpoint_path):
        return False

    if os.path.getsize(checkpoint_path) == 0:
        return False

    return True


def check_weights_in_model(model) -> bool:
    """Check if model has weights loaded.

    Args:
        model: PyTorch model to check.

    Returns:
        True if model has non-empty state dict, False otherwise.
    """
    try:
        state_dict = model.state_dict()
        return any(
            tensor.numel() > 0 for tensor in state_dict.values()
        )
    except (AttributeError, TypeError):
        return False


def validate(
    self,
    model,
    dataloader,
    criterion=None,
    metrics: Optional[List[str]] = None,
    device: Optional[str] = None,
    num_batches: int = -1,
    desc: str = "Validation",
    show_progress: bool = True,
) -> Dict[str, float]:
    """Validate model on a dataset.

    Args:
        model: PyTorch model to validate.
        dataloader: Data loader for validation data.
        criterion: Loss criterion (None = no loss computation).
        metrics: List of metrics to compute (None = use self.metrics).
        device: Device to use (None = use self.device).
        num_batches: Number of batches to validate (None = all).
        desc: Description for progress bar.
        show_progress: Whether to show progress bar.

    Returns:
        Dictionary mapping metric names to their computed values.
    """
    device = device or self.device
    model.to(device)
    model.eval()

    metrics = metrics or self.metrics
    result = {metric: 0.0 for metric in metrics}

    if criterion is not None:
        result["loss"] = 0.0

    num_batches = len(dataloader) if num_batches < 0 else min(num_batches, len(dataloader))

    with torch.no_grad():
        iterator = dataloader
        if show_progress:
            iterator = tqdm(
                dataloader,
                total=num_batches,
                desc=desc,
                unit="batch",
                leave=False,
            )

        for i, (inputs, targets) in enumerate(iterator):
            if i >= num_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            for metric_name in metrics:
                if metric_name == "accuracy":
                    result["accuracy"] += _accuracy(outputs, targets)
                elif metric_name == "loss" and criterion is not None:
                    result["loss"] += _loss(outputs, targets, criterion)

    # Average over number of batches and samples
    num_samples = 0
    for metric_name in metrics:
        if metric_name == "accuracy":
            num_samples = len(dataloader.dataset)
        else:
            num_samples = len(dataloader) * dataloader.batch_size

    for metric_name in result:
        if metric_name in {"accuracy", "loss"}:
            result[metric_name] /= num_batches if num_batches > 0 else 1

    return {k: float(v) for k, v in result.items()}


def validate_with_checkpoint_check(
    self,
    model,
    dataloader,
    checkpoint_manager,
    checkpoint_filename: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """Validate model and check for available checkpoint.

    Args:
        model: PyTorch model to validate.
        dataloader: Data loader for validation data.
        checkpoint_manager: CheckpointManager instance.
        checkpoint_filename: Checkpoint filename to check (None = best.pth).
        **kwargs: Additional arguments to pass to validate().

    Returns:
        Dictionary mapping metric names to their computed values.
    """
    # Check for checkpoint
    filename = checkpoint_filename or "best.pth"
    if checkpoint_manager.checkpoint_exists(filename):
        print(f"Loading weights from: {filename}")
        checkpoint_manager.load_model(model, filename)

    return validate(model, dataloader, **kwargs)


def register_custom_metric(
    self,
    name: str,
    fn: Callable,
    overwrite: bool = False,
) -> None:
    """Register a custom metric function.

    Args:
        name: Name of the metric.
        fn: Callable function that takes (outputs, targets, **kwargs) and returns a float.
        overwrite: Whether to overwrite existing metric with same name.
    """
    if overwrite or name not in _default_metric_functions():
        self.metric_fn[name] = fn


def get_available_metrics(self) -> List[str]:
    """Get list of available metrics.

    Returns:
        List of metric names.
    """
    return list(self.metric_fn.keys())


def clear_metrics(self) -> None:
    """Clear all registered metric functions."""
    self.metric_fn = _default_metric_functions()


def validate_batch(
    self,
    model,
    inputs,
    targets,
    criterion=None,
    **kwargs
) -> Dict[str, float]:
    """Validate on a single batch.

    Args:
        model: PyTorch model to validate.
        inputs: Batch of inputs.
        targets: Batch of targets.
        criterion: Loss criterion.
        **kwargs: Additional arguments to pass to validate().

    Returns:
        Dictionary mapping metric names to their computed values.
    """
    result = {}
    device = self.device

    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        for metric_name, metric_fn in self.metric_fn.items():
            if metric_name == "loss" and criterion is not None:
                result[metric_name] = metric_fn(outputs, targets, criterion)
            else:
                result[metric_name] = metric_fn(outputs, targets)

    return result


def get_device(self) -> str:
    """Get the device being used for validation.

    Returns:
        Device name.
    """
    return self.device


def set_device(self, device: str) -> None:
    """Set the device for validation.

    Args:
        device: Device to use.
    """
    self.device = device


def __repr__(self) -> str:
    return f"Validator(device={self.device}, metrics={self.metrics})"