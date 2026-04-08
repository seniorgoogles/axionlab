"""Configuration writer utilities.

This module provides utilities for writing configuration files in various formats.
"""

import json
import yaml
from typing import Dict, Any, Union, Optional
from pathlib import Path


class ConfigWriter:
    """Write and save configuration files in various formats."""

    @staticmethod
    def write_yaml(config: Union[Dict[str, Any], object], file_path: str) -> None:
        """Write configuration to a YAML file.

        Args:
            config: Configuration dictionary or dataclass to write.
            file_path: Path to output file.
        """
        # Convert dataclass to dict if needed
        if hasattr(config, 'to_dict'):
            config = config.to_dict()

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_json(config: Union[Dict[str, Any], object], file_path: str, indent: int = 2) -> None:
        """Write configuration to a JSON file.

        Args:
            config: Configuration dictionary or dataclass to write.
            file_path: Path to output file.
            indent: JSON indentation level.
        """
        # Convert dataclass to dict if needed
        if hasattr(config, 'to_dict'):
            config = config.to_dict()

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config, f, indent=indent)

    @staticmethod
    def write_model_config(config: object, save_dir: Optional[Union[str, Path]] = None,
                           experiment_name: Optional[str] = None,
                           format: str = "yaml") -> Path:
        """Write model configuration to file.

        Args:
            config: ModelConfig or similar configuration object.
            save_dir: Directory to save config. Defaults to 'experiments/<experiment_name>'.
            experiment_name: Name of experiment. Defaults to timestamp.
            format: Output format ('yaml' or 'json').

        Returns:
            Path to saved configuration file.
        """
        # Convert dataclass to dict if needed
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config

        if save_dir is None and experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if save_dir is None and experiment_name:
            save_dir = Path("experiments") / experiment_name
        else:
            save_dir = Path(save_dir)

        path = save_dir / "config.yaml" if format == "yaml" else save_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            ConfigWriter.write_yaml(config_dict, str(path))
        else:
            ConfigWriter.write_json(config_dict, str(path))

        return path

    @staticmethod
    def write_dict_with_metadata(config: Dict[str, Any],
                                  metadata: Dict[str, Any],
                                  output_file: str) -> None:
        """Write configuration with metadata as a separate file.

        Args:
            config: Main configuration dictionary.
            metadata: Metadata dictionary.
            output_file: Base path for output files.
        """
        # Write main config
        ConfigWriter.write_yaml(config, f"{output_file}.yaml")

        # Write JSON version
        ConfigWriter.write_json(config, f"{output_file}.json")

        # Write metadata as JSON
        with open(f"{output_file}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def write_experiment_config(config: Dict[str, Any],
                                output_dir: str,
                                experiment_name: Optional[str] = None) -> Path:
        """Write experiment configuration with timestamped subdirectory.

        Args:
            config: Experiment configuration dictionary.
            output_dir: Output directory.
            experiment_name: Optional experiment name.

        Returns:
            Path to the configuration file.
        """
        from datetime import datetime

        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        output_path = Path(output_dir) / experiment_name / "config.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ConfigWriter.write_yaml(config, str(output_path))

        return output_path

    @staticmethod
    def append_to_yaml(config_file: str, new_config: Dict[str, Any],
                       merge_key: Optional[str] = None) -> None:
        """Append or merge configuration into existing YAML file.

        Args:
            config_file: Path to existing YAML file.
            new_config: New configuration to append/merge.
            merge_key: If provided, merge nested dict under this key.
        """
        path = Path(config_file)

        # Load existing config
        existing_config = ConfigReader.read_config(config_file, validate=False)

        # Merge config
        if merge_key and merge_key in existing_config:
            # Merge nested dictionaries
            if isinstance(existing_config[merge_key], dict) and isinstance(new_config.get(merge_key), dict):
                existing_config[merge_key].update(new_config[merge_key])
            else:
                existing_config[merge_key] = new_config[merge_key]
        else:
            # Append or override at top level
            existing_config.update(new_config)

        # Write back
        ConfigWriter.write_yaml(existing_config, config_file)