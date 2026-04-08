"""Utility functions and classes.

This module provides common utilities used across the library.

Components:
- DeviceSelector: Automatic device selection (CUDA, MPS, CPU)
- Mapper: Configuration mapping utilities for layers and quantizers
- timer: Decorator for timing function execution
"""

from .device_selector import DeviceSelector
from .mapper import Mapper
from .timer import timer

__all__ = [
    "DeviceSelector",
    "Mapper",
    "timer",
]
