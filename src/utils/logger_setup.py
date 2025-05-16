# src/utils/logger_setup.py

import logging
import logging.config
import yaml
import os

DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        # Basic file handler, can be expanded in logging_config.yaml
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'standard',
            'filename': 'logs/a2ir_run.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 3,
            'level': 'DEBUG',
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    }
}

def setup_logging(config_path: str = "config/logging_config.yaml", default_level=logging.INFO):
    """Sets up logging from a YAML file or uses a default."""
    # Ensure logs directory exists
    log_file_path = DEFAULT_LOGGING_CONFIG.get('handlers', {}).get('file', {}).get('filename', 'logs/a2ir_run.log')
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            # Fallback to basic config if directory creation fails
            logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.error(f"Failed to create log directory {log_dir}: {e}. Using basicConfig.", exc_info=True)
            return


    effective_config = DEFAULT_LOGGING_CONFIG
    loaded_from_file = False

    if os.path.exists(config_path):
        try:
            with open(config_path, 'rt') as f:
                file_config = yaml.safe_load(f.read())
            if file_config: # Check if file_config is not None or empty
                effective_config = file_config # Use file config if valid
                # Update log file path from file_config if specified
                file_handler_path = effective_config.get('handlers', {}).get('file', {}).get('filename')
                if file_handler_path:
                    new_log_dir = os.path.dirname(file_handler_path)
                    if new_log_dir and not os.path.exists(new_log_dir):
                         os.makedirs(new_log_dir, exist_ok=True)
                loaded_from_file = True
            logging.config.dictConfig(effective_config)
            if loaded_from_file:
                logging.getLogger(__name__).info(f"Logging configured from {config_path}.")
            else:
                logging.getLogger(__name__).info(f"Logging config file {config_path} was empty or invalid. Using default logging setup.")

        except Exception as e:
            # Fallback to basicConfig if dictConfig fails with file content
            logging.basicConfig(level=default_level, format='%(asctime)s - %(levelname)s - %(message)s')
            logging.getLogger(__name__).error(f"Error applying logging config from {config_path}: {e}. Using basicConfig.", exc_info=True)
    else:
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        logging.getLogger(__name__).info(f"Logging config file {config_path} not found. Using default logging setup.")

