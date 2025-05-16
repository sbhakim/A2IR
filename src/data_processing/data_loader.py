# src/data_processing/data_loader.py
import pandas as pd
import os
import logging
from typing import Dict, Any, Optional, List
# Import the new preprocessor and the general one
from .alert_preprocessor import AlertPreprocessor
from .dataset_preprocessors.cic_ids_preprocessor import CICIDSPreprocessor  # New import

logger = logging.getLogger(__name__)


def _load_and_preprocess_single_dataset(dataset_name: str, config: Dict[str, Any],
                                        alert_standardizer: AlertPreprocessor) -> Optional[
    List[Dict[str, Any]]]:  # Return List of dicts
    """
    Loads and preprocesses a single specified dataset based on its configuration.
    This is a dispatcher function.
    """
    logger.info(f"Attempting to load and preprocess dataset: {dataset_name}")
    dataset_cfg = config.get('datasets', {}).get(dataset_name)

    if not dataset_cfg:
        logger.error(f"Configuration for dataset '{dataset_name}' not found.")
        return None

    raw_path_config = dataset_cfg.get('path')
    preprocessing_steps = dataset_cfg.get('preprocessing', {})  # You might use this later

    if not raw_path_config:
        logger.error(f"Path for dataset '{dataset_name}' not specified in configuration.")
        return None

    absolute_raw_path = os.path.abspath(raw_path_config)
    logger.info(f"Dataset '{dataset_name}': Path='{absolute_raw_path}', Preprocessing Steps='{preprocessing_steps}'")

    processed_alerts: List[Dict[str, Any]] = []

    if dataset_name == "cic_ids2017":
        if not os.path.exists(absolute_raw_path):
            logger.error(f"Path does not exist for CIC-IDS2017: {absolute_raw_path}")
            return None
        cic_preprocessor = CICIDSPreprocessor(alert_standardizer=alert_standardizer)
        if os.path.isdir(absolute_raw_path):
            processed_alerts = cic_preprocessor.process_dataset_directory(absolute_raw_path)
        elif os.path.isfile(absolute_raw_path) and absolute_raw_path.lower().endswith('.csv'):
            processed_alerts = cic_preprocessor.load_and_process_cic_ids(absolute_raw_path)
        else:
            logger.error(f"Path for CIC-IDS2017 is not a valid CSV file or directory: {absolute_raw_path}")
            return None

    elif dataset_name == "lanl_logs":  # Example for another dataset
        logger.warning(
            f"Placeholder: Loading and preprocessing for {dataset_name} needs full implementation using alert_standardizer.")
        # Example: Find first .txt or .csv file in the directory
        # file_to_load = None
        # if os.path.isdir(absolute_raw_path):
        #    for f_name in os.listdir(absolute_raw_path):
        #        if f_name.endswith(('.txt', '.csv')):
        #            file_to_load = os.path.join(absolute_raw_path, f_name)
        #            break
        # elif os.path.isfile(absolute_raw_path):
        #    file_to_load = absolute_raw_path
        # if file_to_load:
        #    processed_alerts = alert_standardizer.preprocess_file(file_to_load) # Use the general preprocessor
        # else:
        #    logger.error(f"No suitable data file found for LANL: {absolute_raw_path}")
        # For testing, let's create dummy data
        processed_alerts = [alert_standardizer.standardize_alert(
            {'id': 'lanl_dummy_1', 'message': 'dummy lanl log entry', '_time': pd.Timestamp.now().timestamp()}, 'lanl')]

    # Add other dataset handlers here:
    # elif dataset_name == "splunk_attack_range":
    #    processed_alerts = alert_standardizer.preprocess_file(absolute_raw_path)

    else:
        logger.error(f"No specific loader implemented for dataset: {dataset_name}")
        return None

    if processed_alerts:
        logger.info(
            f"Successfully loaded and performed initial processing for {dataset_name}. Num alerts: {len(processed_alerts)}")
        # Optionally save processed alerts
        processed_dir = os.path.join("data", "processed", dataset_name)
        os.makedirs(processed_dir, exist_ok=True)
        output_file_path = os.path.join(processed_dir, f"{dataset_name}_processed_alerts.json")
        try:
            import json
            with open(output_file_path, 'w') as f:
                json.dump(processed_alerts, f, indent=2)
            logger.info(f"Saved processed {dataset_name} alerts to {output_file_path}")
        except Exception as e:
            logger.error(f"Could not save processed alerts for {dataset_name}: {e}")

    return processed_alerts


def run_all_preprocessing(config: Dict[str, Any]):
    """
    Iterates through all datasets defined in the configuration and preprocesses them.
    """
    datasets_config = config.get('datasets')
    if not datasets_config:
        logger.error("No 'datasets' section found in the main configuration.")
        return

    # Initialize the general alert standardizer once
    general_alert_standardizer = AlertPreprocessor(config.get('preprocessing', {}))

    logger.info("Starting preprocessing for all configured datasets.")
    all_processed_data = {}
    for dataset_name in datasets_config.keys():
        processed_data = _load_and_preprocess_single_dataset(dataset_name, config, general_alert_standardizer)
        if processed_data:
            all_processed_data[dataset_name] = processed_data
            logger.info(f"Finished preprocessing for {dataset_name}. Alerts: {len(processed_data)}")
        else:
            logger.warning(f"Preprocessing returned no data for {dataset_name}.")

    logger.info("Finished preprocessing for all configured datasets.")
    return all_processed_data  # Return all processed data if needed by orchestrator