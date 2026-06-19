"""Core: model building and the training/eval Runner."""

from src.core.runner import Phase, Runner, build_network

__all__ = ["Runner", "Phase", "build_network",
           "quantize_model", "set_bit_width", "run_qat_schedule", "QatStage"]

# lazy so importing the Runner doesn't require brevitas
_LAZY = {
    "quantize_model": "src.core.quantize",
    "set_bit_width": "src.core.quantize",
    "run_qat_schedule": "src.core.qat_schedule",
    "QatStage": "src.core.qat_schedule",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
