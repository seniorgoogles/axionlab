import yaml
import copy
from typing import Dict, Any
from pathlib import Path


class ConfigReader:
    """Read and merge YAML configuration files with inheritance support."""

    @staticmethod
    def read_config(config_file: str, validate: bool = True) -> Dict[str, Any]:
        """Read configuration from a file with inheritance support.

        Args:
            config_file: Path to the configuration file.
            validate: Whether to validate the final merged config.

        Returns:
            Configuration dictionary with inheritance resolved.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file has invalid YAML syntax.
            ValueError: If config file is empty or invalid.
        """
        return ConfigReader._read_config_with_inheritance(config_file, validate)

    @staticmethod
    def read_validated_config(config_file: str):
        """Read and validate configuration, returning structured config.

        Args:
            config_file: Path to configuration file

        Returns:
            Validated ModelConfig instance
        """
        from .config_validator import ConfigValidator
        config_dict = ConfigReader.read_config(config_file, validate=False)
        return ConfigValidator.validate_config(config_dict)

    @staticmethod
    def _read_config_with_inheritance(config_file: str, validate: bool = True) -> Dict[str, Any]:
        """Read configuration with inheritance support.

        The config file can specify a 'base' field pointing to another config file.
        The base config is loaded first, then the child config overrides/extends it.
        Supports multiple levels of inheritance.

        Args:
            config_file: Path to the configuration file.
            validate: Whether to validate the final merged config.

        Returns:
            Configuration dictionary with inheritance resolved.
        """
        if not config_file:
            raise ValueError("Config file path cannot be empty")

        config_path = Path(config_file)

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ValueError(f"Config file '{config_file}' is empty or invalid")

            # Handle inheritance
            if 'base' in config:
                base_path = config['base']

                # Resolve relative paths relative to current config file
                if not Path(base_path).is_absolute():
                    base_path = str(config_path.parent / base_path)

                # Load base config recursively
                base_config = ConfigReader._read_config_with_inheritance(base_path, validate=False)

                # Merge configs (child overrides parent)
                merged_config = ConfigReader._deep_merge(base_config, config)

                # Remove the 'base' key from final config
                if 'base' in merged_config:
                    del merged_config['base']

                config = merged_config

            if validate:
                from .config_validator import ConfigValidator
                ConfigValidator.validate_config(config)

            return config

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML syntax in {config_file}: {e}")

    @staticmethod
    def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries with protected fields support.

        The base_dict can contain a '_protected_fields' key listing fields
        that should not be overridden by the child config.

        Args:
            base_dict: Base dictionary (may contain '_protected_fields')
            override_dict: Dictionary with override values

        Returns:
            Merged dictionary with protected fields preserved
        """
        result = copy.deepcopy(base_dict)

        # Get protected fields list from base config
        protected_fields = result.pop('_protected_fields', [])

        for key, value in override_dict.items():
            # Skip protected fields
            if key in protected_fields:
                continue

            # Skip the _protected_fields key from override
            if key == '_protected_fields':
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigReader._deep_merge(result[key], value)
            else:
                # Override completely (including lists)
                result[key] = copy.deepcopy(value)

        return result

    @staticmethod
    def write_config(config: Dict[str, Any], file_path: str) -> None:
        """Write configuration to a YAML file.

        Args:
            config: Configuration dictionary to write
            file_path: Path to output file
        """
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
