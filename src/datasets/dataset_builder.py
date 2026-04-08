import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any

from src.core.inject.enum import DatasetTypes
from src.datasets.imagenet import ImageNet
from src.datasets.jetSubstructure.dataloader import JetSubstructureDataset
from src.datasets.mnist import Mnist
from src.datasets.fashionMnist import FashionMnist
from src.utils.mapper import Mapper
import yaml


@dataclass
class DatasetConfig:
    """Configuration for dataset initialization parameters."""
    dataset_class: type
    init_params: Dict[str, Any]
    requires_config_param: Optional[str] = None  # e.g., 'dataset_root_path' for JSC


class DatasetRegistry:
    """Registry for dataset types to their corresponding classes and configurations."""

    _registry: Dict[DatasetTypes, DatasetConfig] = {}

    @classmethod
    def register(
        cls,
        dataset_type: DatasetTypes,
        dataset_class: type,
        **init_kwargs
    ):
        """Register a dataset type with its class and initialization parameters."""
        cls._registry[dataset_type] = DatasetConfig(
            dataset_class=dataset_class,
            init_params=init_kwargs
        )

    @classmethod
    def get_config(cls, dataset_type: DatasetTypes) -> Optional[DatasetConfig]:
        """Get the configuration for a dataset type."""
        return cls._registry.get(dataset_type)

    @classmethod
    def build_dataset(cls, dataset_type: DatasetTypes, config: Dict[str, Any], **extra_kwargs):
        """Build and instantiate a dataset using its registered configuration."""
        dataset_config = cls._registry.get(dataset_type)

        if dataset_config is None:
            raise ValueError(f"Dataset type {dataset_type} not registered")

        # Extract required config parameter if specified
        if dataset_config.requires_config_param:
            required_param = dataset_config.requires_config_param
            if required_param not in config:
                raise ValueError(
                    f"Config missing required parameter '{required_param}' "
                    f"for dataset type {dataset_type}"
                )

            # Add required parameter to init params
            dataset_config.init_params[required_param] = config[required_param]

        # Call the dataset constructor with all init params and extra kwargs
        return dataset_config.dataset_class(
            **dataset_config.init_params,
            **extra_kwargs
        )


class DatasetBuilder:
    """Factory for building dataset instances."""

    # Register datasets
    @staticmethod
    def _register_datasets():
        """Register all known dataset types."""
        DatasetRegistry.register(
            DatasetTypes.MNIST,
            Mnist,
            crop_border_pixels=0
        )

        DatasetRegistry.register(
            DatasetTypes.FASHION_MNIST,
            FashionMnist
        )

        DatasetRegistry.register(
            DatasetTypes.IMAGENET,
            ImageNet
        )

        DatasetRegistry.register(
            DatasetTypes.JSC,
            JetSubstructureDataset,
            requires_config_param='dataset_root_path'
        )

    @staticmethod
    def build(dataset, config, crop_border_pixels=0):
        """
        Build a dataset instance.

        Args:
            dataset: DatasetTypes enum value or string
            config: Configuration dictionary or path to YAML file
            crop_border_pixels: Border crop pixels (for MNIST only)

        Returns:
            Dataset instance
        """
        # Initialize registry with known datasets
        DatasetBuilder._register_datasets()

        # Convert dataset to enum if string
        if isinstance(dataset, str):
            try:
                dataset = DatasetTypes[dataset.upper()]
            except KeyError:
                raise ValueError(f"Unknown dataset type: {dataset}")

        # Load config if path provided
        config_path = config if isinstance(config, str) else None
        if not isinstance(config, dict):
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
            else:
                raise Exception("Config does not exist")

        train_path = config["train_path"]
        test_path = config["test_path"]

        batch_size = config["batch_size"]
        num_workers = config["num_workers"]
        distributed = config["distributed"]

        # Build and return the appropriate dataset
        if dataset == DatasetTypes.JSC:
            return DatasetRegistry.build_dataset(
                dataset,
                config,
                batch_size=batch_size,
                distributed=distributed,
                num_workers=num_workers
            )
        else:
            # For other datasets, pass extra kwargs
            return DatasetRegistry.build_dataset(
                dataset,
                config,
                train_path=train_path,
                test_path=test_path,
                batch_size=batch_size,
                distributed=distributed,
                num_workers=num_workers,
                crop_border_pixels=crop_border_pixels if dataset == DatasetTypes.MNIST else 0
            )