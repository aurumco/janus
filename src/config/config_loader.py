"""Configuration loader module for managing application settings."""

from pathlib import Path
from typing import Any, Dict, Optional

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

        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing configuration parameters.
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

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
