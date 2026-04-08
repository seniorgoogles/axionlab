import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LayerDefinition:
    """Definition for a single layer in the model."""
    from_layer: int  # Input source (-1 for previous layer)
    repeats: int  # Number of repetitions
    module_type: str  # PyTorch module type (Linear, Conv2d, etc.)
    name: str  # Layer name
    args: Dict[str, Any]  # Layer arguments


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    root_path: Optional[str] = None
    num_workers: int = 4
    batch_size: List[int] = field(default_factory=lambda: [32, 32])
    distributed: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    weight_decay: float = 0.0


@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    enabled: bool = False
    weight_bit_width: int = 8
    act_bit_width: int = 8
    bias_bit_width: int = 32


@dataclass
class ModelConfig:
    """Complete model configuration."""
    name: str
    dataset: DatasetConfig
    training: TrainingConfig
    num_classes: int
    backbone: List[LayerDefinition]
    neck: Optional[List[LayerDefinition]] = None
    head: Optional[List[LayerDefinition]] = None
    quantization: Optional[QuantizationConfig] = None
    weights_path: Optional[str] = None

    @property
    def all_layers(self) -> List[LayerDefinition]:
        """Get all layers in order: backbone + neck + head."""
        layers = list(self.backbone)
        if self.neck:
            layers.extend(self.neck)
        if self.head:
            layers.extend(self.head)
        return layers


class ConfigValidator:
    """Validates and normalizes configuration files for axionlab."""

    # Supported layer types
    SUPPORTED_LAYERS = {
        # Standard PyTorch layers
        'Linear', 'Conv2d', 'Conv1d', 'ConvTranspose2d',
        'BatchNorm2d', 'BatchNorm1d', 'LayerNorm', 'GroupNorm',
        'ReLU', 'LeakyReLU', 'GELU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
        'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'AdaptiveMaxPool2d',
        'Dropout', 'Dropout2d',
        'Flatten', 'Unflatten',
        # Quantized layers (Brevitas)
        'QuantLinear', 'QuantConv2d', 'QuantReLU', 'QuantIdentity',
        # Custom layers
        'BasicBlock', 'Bottleneck',
    }

    # Required arguments per layer type
    LAYER_REQUIREMENTS = {
        'Linear': ['in_features', 'out_features'],
        'Conv2d': ['in_channels', 'out_channels', 'kernel_size'],
        'Conv1d': ['in_channels', 'out_channels', 'kernel_size'],
        'BatchNorm2d': ['num_features'],
        'BatchNorm1d': ['num_features'],
        'MaxPool2d': ['kernel_size'],
        'AvgPool2d': ['kernel_size'],
        'AdaptiveAvgPool2d': ['output_size'],
        'QuantLinear': ['in_features', 'out_features'],
        'QuantConv2d': ['in_channels', 'out_channels', 'kernel_size'],
    }

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> ModelConfig:
        """Validate and convert config dict to structured format.

        Args:
            config: Raw configuration dictionary

        Returns:
            Validated ModelConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        # Required top-level fields
        required_fields = ['name', 'backbone']
        missing = [f for f in required_fields if f not in config]
        if missing:
            raise ValueError(f"Missing required configuration fields: {missing}")

        # Parse dataset config
        dataset_config = ConfigValidator._parse_dataset_config(config)

        # Parse training config
        training_config = ConfigValidator._parse_training_config(config)

        # Parse quantization config if present
        quant_config = None
        if 'quantization' in config:
            quant_config = ConfigValidator._parse_quantization_config(config['quantization'])

        # Parse layer definitions
        backbone = ConfigValidator._parse_layers(config['backbone'], 'backbone')
        neck = None
        head = None

        if 'neck' in config:
            neck = ConfigValidator._parse_layers(config['neck'], 'neck')
        if 'head' in config:
            head = ConfigValidator._parse_layers(config['head'], 'head')

        # Get number of classes
        num_classes = config.get('nc', config.get('num_classes', 10))

        logger.info("Configuration validation passed")

        return ModelConfig(
            name=config['name'],
            dataset=dataset_config,
            training=training_config,
            num_classes=num_classes,
            backbone=backbone,
            neck=neck,
            head=head,
            quantization=quant_config,
            weights_path=config.get('weights_path'),
        )

    @staticmethod
    def _parse_dataset_config(config: Dict[str, Any]) -> DatasetConfig:
        """Parse dataset configuration from config dict."""
        batch_size = config.get('batch_size', [32, 32])
        if isinstance(batch_size, int):
            batch_size = [batch_size, batch_size]

        return DatasetConfig(
            name=config.get('dataset', 'unknown'),
            train_path=config.get('train_path'),
            test_path=config.get('test_path'),
            root_path=config.get('dataset_root_path'),
            num_workers=config.get('num_workers', 4),
            batch_size=batch_size,
            distributed=config.get('distributed', False),
        )

    @staticmethod
    def _parse_training_config(config: Dict[str, Any]) -> TrainingConfig:
        """Parse training configuration from config dict."""
        return TrainingConfig(
            epochs=config.get('epochs', 100),
            learning_rate=config.get('learning_rate', config.get('lr', 0.001)),
            optimizer=config.get('optimizer', 'adam'),
            scheduler=config.get('scheduler'),
            weight_decay=config.get('weight_decay', 0.0),
        )

    @staticmethod
    def _parse_quantization_config(quant_dict: Dict[str, Any]) -> QuantizationConfig:
        """Parse quantization configuration."""
        return QuantizationConfig(
            enabled=quant_dict.get('enabled', True),
            weight_bit_width=quant_dict.get('weight_bit_width', 8),
            act_bit_width=quant_dict.get('act_bit_width', 8),
            bias_bit_width=quant_dict.get('bias_bit_width', 32),
        )

    @staticmethod
    def _parse_layers(layers_config: List, section_name: str) -> List[LayerDefinition]:
        """Parse layer definitions from config.

        Supports two formats:
        1. List format: [from, repeats, module_type, name, args]
        2. Dict format for nested layers (BasicBlock, etc.)
        """
        layers = []

        for idx, layer_def in enumerate(layers_config):
            if isinstance(layer_def, list) and len(layer_def) >= 4:
                # Standard format: [from, repeats, module_type, name, args]
                from_layer = layer_def[0]
                repeats = layer_def[1]
                module_type = layer_def[2]
                name = layer_def[3]
                args = layer_def[4] if len(layer_def) > 4 else {}

                # Validate layer type
                if module_type not in ConfigValidator.SUPPORTED_LAYERS:
                    logger.warning(f"Layer type '{module_type}' at {section_name}[{idx}] "
                                   f"not in standard supported list")

                # Validate required arguments
                ConfigValidator._validate_layer_args(module_type, args, f"{section_name}[{idx}]")

                layers.append(LayerDefinition(
                    from_layer=from_layer,
                    repeats=repeats,
                    module_type=module_type,
                    name=name,
                    args=args,
                ))

            elif isinstance(layer_def, dict):
                # Nested layer format (e.g., BasicBlock with sub-layers)
                # Store as-is for now, model builder will handle
                for block_name, block_def in layer_def.items():
                    layers.append(LayerDefinition(
                        from_layer=-1,
                        repeats=1,
                        module_type='Block',
                        name=block_name,
                        args={'definition': block_def},
                    ))
            else:
                raise ValueError(f"Invalid layer definition at {section_name}[{idx}]: {layer_def}")

        return layers

    @staticmethod
    def _validate_layer_args(layer_type: str, args: Dict[str, Any], location: str) -> None:
        """Validate layer arguments against requirements."""
        if layer_type in ConfigValidator.LAYER_REQUIREMENTS:
            required = ConfigValidator.LAYER_REQUIREMENTS[layer_type]
            missing = [arg for arg in required if arg not in args]
            if missing:
                raise ValueError(f"Layer at {location} ({layer_type}) missing required args: {missing}")

        # Type-specific validations
        if layer_type in ('Linear', 'QuantLinear'):
            if 'in_features' in args and args['in_features'] <= 0:
                raise ValueError(f"Layer at {location}: in_features must be positive")
            if 'out_features' in args and args['out_features'] <= 0:
                raise ValueError(f"Layer at {location}: out_features must be positive")

        elif layer_type in ('Conv2d', 'QuantConv2d', 'Conv1d'):
            if 'in_channels' in args and args['in_channels'] <= 0:
                raise ValueError(f"Layer at {location}: in_channels must be positive")
            if 'out_channels' in args and args['out_channels'] <= 0:
                raise ValueError(f"Layer at {location}: out_channels must be positive")

        elif layer_type == 'Dropout':
            if 'p' in args and not (0 <= args['p'] <= 1):
                raise ValueError(f"Layer at {location}: Dropout p must be between 0 and 1")

    @staticmethod
    def validate_raw(config: Dict[str, Any]) -> bool:
        """Quick validation without parsing to dataclasses.

        Args:
            config: Raw configuration dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        required = ['name', 'backbone']
        missing = [f for f in required if f not in config]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        if not isinstance(config['backbone'], list):
            raise ValueError("backbone must be a list of layer definitions")

        if len(config['backbone']) == 0:
            raise ValueError("backbone cannot be empty")

        return True
