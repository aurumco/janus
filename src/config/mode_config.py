"""Mode-specific configuration wrapper."""

from typing import Any, Optional
from src.config.config_loader import ConfigLoader


class ModeConfig:
    """Helper class to fetch config with mode prefix first, then fall back to global."""

    def __init__(self, full_cfg: ConfigLoader, prefix: str) -> None:
        """Initialize the ModeConfig.

        Args:
            full_cfg: The full ConfigLoader instance.
            prefix: The prefix to check first (e.g., 'pretrain' or 'finetune').
        """
        self.full_cfg = full_cfg
        self.prefix = prefix

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Fetch config with mode prefix first, then fall back to global.

        Example: if prefix='pretrain' and key='data.path', this checks
        'pretrain.data.path' first, then 'data.path'.

        Args:
            key: Config key.
            default: Default value if not found.

        Returns:
            The configuration value.
        """
        mode_key = f"{self.prefix}.{key}"
        mode_val = self.full_cfg.get(mode_key, None)
        if mode_val is not None:
            return mode_val
        return self.full_cfg.get(key, default)
