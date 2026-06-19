"""Utility functions and classes.

Components:
- DeviceSelector: Automatic device selection (CUDA/ROCm, MPS, CPU)
- timer: Decorator for timing function execution
"""

from .device_selector import DeviceSelector
from .timer import timer

__all__ = [
    "DeviceSelector",
    "timer",
]
