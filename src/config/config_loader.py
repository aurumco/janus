"""Configuration loader module for managing application settings."""

from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    def __init__(self, config_path: str) -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the configuration file is invalid.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._config = self._load_config(self.config_path)

    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file, handling inheritance.

        Args:
            path: Path to the config file.

        Returns:
            Dictionary containing configuration parameters.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        # Handle inheritance
        defaults = config.get("defaults", [])
        if defaults:
            base_config = {}
            # Allow defaults to be a list of strings or dicts (though usually strings)
            for default_file in defaults:
                if isinstance(default_file, str):
                    # Resolve relative path
                    default_path = path.parent / default_file
                    if not default_path.exists():
                        # Try relative to cwd if not found relative to config
                        default_path = Path(default_file)

                    if default_path.exists():
                        base_data = self._load_config(default_path)
                        self._deep_merge(base_config, base_data)
                    else:
                        print(f"Warning: Default config {default_file} not found.")

            # Merge current config over base
            self._deep_merge(base_config, config)
            config = base_config

        return config

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively merge update dict into base dict."""
        for key, value in update.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value by key.

        Args:
            key: Dot-separated key path (e.g., 'model.d_model').
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section.

        Args:
            section: Name of the configuration section.

        Returns:
            Dictionary containing the section configuration.
        """
        return self._config.get(section, {})

    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.

        Returns:
            Complete configuration dictionary.
        """
        return self._config
