"""Model registry for model types.

This module provides a registry pattern for model types, allowing
registration and retrieval of model classes for factory functions.
"""

from typing import Dict, Type, Optional, Any, Union
from abc import ABC

from src.models.base import BaseModel, _get_model_info_from_class


class ModelRegistry:
    """Registry for model types and their corresponding classes.

    The registry provides a centralized location to register model classes
    and retrieve them by name or type.

    Attributes:
        _registry: Dictionary mapping model names/types to their classes.

    Example:
        >>> @ModelRegistry.register("my_model")
        >>> class MyModel(BaseModel):
        ...     def forward(self, x):
        ...         return x
        >>> model = build_model("my_model")
    """

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        info: Optional[str] = None
    ):
        """Decorator to register a model class.

        Args:
            name: Name to register the model under. If None, uses class name.
            info: Description for the model.

        Returns:
            Decorator function.
        """
        def decorator(cls: Type[BaseModel]):
            # Generate name if not provided
            model_name = name or cls.__name__
            cls._registry_name = model_name

            # Register in global registry
            cls._registry[model_name] = cls

            # Store info for documentation
            if info:
                cls._registry_info = info

            return cls
        return decorator

    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        """Get model class by name.

        Args:
            name: Name of the model class.

        Returns:
            Model class.

        Raises:
            KeyError: If model name not found.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Model '{name}' not found. Available models: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names.

        Returns:
            List of registered model names.
        """
        return list(cls._registry.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model.

        Args:
            name: Name of the model.

        Returns:
            Dictionary with model information, or None if not found.
        """
        model_class = cls._registry.get(name)
        if model_class is None:
            return None

        return {
            "name": name,
            "class": model_class,
            "module": model_class.__module__,
            "info": getattr(model_class, "_registry_info", None),
            "input_shape": getattr(model_class, "input_shape", None),
            "output_shape": getattr(model_class, "output_shape", None),
        }

    @classmethod
    def get_all_models_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered models.

        Returns:
            Dictionary mapping model names to their information.
        """
        return {
            name: cls.get_model_info(name)
            for name in cls.list_models()
        }


def build_model(
    model_type: Union[str, Type[BaseModel]],
    config: Optional[Any] = None,
    **kwargs
) -> BaseModel:
    """Factory function to build a model instance.

    Args:
        model_type: Model name (string) or model class.
        config: Optional configuration for model initialization.
        **kwargs: Additional arguments to pass to model constructor.

    Returns:
        Model instance.

    Example:
        >>> model = build_model("lenet5", num_classes=10)
        >>> model = build_model(LeNet5, input_shape=(1, 28, 28))
    """
    from src.models.config import ModelConfig

    # If config is provided and is a ModelConfig, use its model_type
    if isinstance(config, ModelConfig):
        model_class = config.model_type
        num_classes = config.num_classes
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = num_classes
    else:
        # Try to resolve model_type
        if isinstance(model_type, str):
            model_class = ModelRegistry.get_model_class(model_type)
        else:
            model_class = model_type

    # Create model instance
    model = model_class(**kwargs)

    return model


# Auto-register all model classes found in src.models
def _register_models_from_module(module_name: str):
    """Register all model classes found in a module.

    Args:
        module_name: Name of the module to search.
    """
    import importlib
    try:
        module = importlib.import_module(module_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseModel) and
                attr is not BaseModel):
                # Register without name (class will use its name)
                ModelRegistry.register(attr)

    except ImportError:
        pass


# Import and register models from common modules
_register_models_from_module("src.models")