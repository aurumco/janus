"""Configuration loader for analysis pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_data_path() -> str:
    """Extract data path from config.
    
    Returns:
        Path to Parquet dataset
    """
    config = load_config()
    
    # Try pretrain path first, fallback to other paths
    if 'pretrain' in config and 'data' in config['pretrain']:
        path = config['pretrain']['data'].get('path', '')
        if path:
            return path
    
    # Fallback: check other sections
    if 'finetune' in config and 'data' in config['finetune']:
        path = config['finetune']['data'].get('path', '')
        if path:
            return path
    
    raise ValueError("No data path found in config.yaml")


def get_feature_columns() -> list:
    """Get list of feature columns from config or use default.
    
    Returns:
        List of feature column names
    """
    # Default features based on typical crypto dataset
    default_features = [
        'RSI_14_M15', 'RSI_14_H1', 'RSI_14_H4',
        'ATR_5_pct_M15', 'ATR_5_pct_H1',
        'ADX_14_M15', 'ADX_14_H1',
        'OBV_M15', 'OBV_H1',
        'dist_from_ema_200_pct_M15', 'dist_from_ema_200_pct_H1',
        'ema_slope_50_M15', 'ema_slope_50_H1',
        'volume_oscillator_M15',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
    ]
    
    config = load_config()
    
    # Check if features are specified in config
    if 'pretrain' in config and 'data' in config['pretrain']:
        features = config['pretrain']['data'].get('feature_columns', default_features)
        return features
    
    return default_features


if __name__ == "__main__":
    print("Data Path:", get_data_path())
    print("Feature Columns:", get_feature_columns())
