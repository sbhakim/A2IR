# src/config_management/config_loader.py

import yaml
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return {}

def load_main_config(main_config_path: str = "config/main_config.yaml") -> Dict[str, Any]:
    """Loads the main application configuration."""
    main_cfg = load_yaml_config(main_config_path)
    # Example: Load other specific configs if needed
    # logging_cfg_path = main_cfg.get('logging_config_path', 'config/logging_config.yaml')
    # main_cfg['logging_settings'] = load_yaml_config(logging_cfg_path)
    return main_cfg
