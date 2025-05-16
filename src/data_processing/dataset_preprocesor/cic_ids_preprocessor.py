# src/data_processing/dataset_preprocessors/cic_ids_preprocessor.py
import pandas as pd
import logging
import os
from typing import List, Dict, Any
from ..alert_preprocessor import \
    AlertPreprocessor  # Assuming AlertPreprocessor can be a base or used for final standardization

logger = logging.getLogger(__name__)


class CICIDSPreprocessor:
    """
    Preprocessor for the CIC-IDS2017 dataset.
    Handles loading, feature engineering, and transformation into A²IR standardized alert format.
    """

    def __init__(self, alert_standardizer: AlertPreprocessor):
        self.alert_standardizer = alert_standardizer
        # Define expected columns (can be loaded from config or schema file)
        # These are just a few examples, refer to the CIC-IDS2017 paper for all features
        self.EXPECTED_COLUMNS = [
            'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port',
            'Protocol', 'Timestamp', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Fwd PSH Flags', 'SYN Flag Count', 'ACK Flag Count', 'URG Flag Count',  # Example flag features
            'Packet Length Mean', 'Average Packet Size', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Active Mean', 'Idle Mean',
            'Label'  # This is crucial for supervised learning and evaluation
        ]
        # Mapping from CIC-IDS2017 labels to A²IR alert types
        self.LABEL_TO_A2IR_TYPE = {
            'BENIGN': 'benign_traffic',
            'DoS Hulk': 'ddos',
            'PortScan': 'reconnaissance',  # Or a more specific 'port_scan' type
            'DDoS': 'ddos',
            'DoS GoldenEye': 'ddos',
            'FTP-Patator': 'brute_force',
            'SSH-Patator': 'brute_force',
            'DoS slowloris': 'ddos_slow',
            'DoS Slowhttptest': 'ddos_slow',
            'Bot': 'botnet_activity',
            'Web Attack  Brute Force': 'web_attack_brute_force',
            'Web Attack  XSS': 'web_attack_xss',
            'Web Attack  Sql Injection': 'web_attack_sqli',
            'Infiltration': 'infiltration',
            'Heartbleed': 'exploit_heartbleed'  # Example
        }

    def load_and_process_cic_ids(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a single CIC-IDS2017 CSV file, processes it, and standardizes alerts.
        """
        logger.info(f"Loading CIC-IDS2017 data from: {file_path}")
        try:
            # CIC-IDS2017 CSVs often have leading/trailing spaces in column names
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()  # Clean column names
            logger.info(f"Successfully loaded {os.path.basename(file_path)}. Shape: {df.shape}")

            # Basic validation for expected columns
            missing_cols = [col for col in self.EXPECTED_COLUMNS if
                            col not in df.columns and col != 'Flow ID']  # Flow ID might not be needed for alert
            if 'Label' not in df.columns:
                logger.error(f"Critical 'Label' column missing in {file_path}. Cannot process for training/evaluation.")
                return []
            if missing_cols:
                logger.warning(
                    f"Missing some expected columns in {file_path}: {missing_cols}. Proceeding with available data.")

        except Exception as e:
            logger.error(f"Error loading CIC-IDS2017 CSV {file_path}: {e}", exc_info=True)
            return []

        standardized_alerts = []
        for index, row in df.iterrows():
            try:
                raw_alert_details = row.to_dict()
                a2ir_alert_type = self.LABEL_TO_A2IR_TYPE.get(raw_alert_details.get('Label', '').strip(),
                                                              'unknown_network_event')

                # Construct a message or structured data for the AlertPreprocessor
                # This part needs careful mapping from CIC-IDS features to what your A²IR expects
                message_parts = [f"{k.replace(' ', '_')}:{v}" for k, v in raw_alert_details.items() if
                                 pd.notna(v) and k != 'Label']
                message = "; ".join(message_parts)

                # Prepare a raw_alert structure that AlertPreprocessor can understand
                # or directly create the standardized A²IR alert structure here.
                # For now, let's create a dictionary that `standardize_alert` can work with.
                raw_alert_for_a2ir = {
                    'id': f"cic_{os.path.basename(file_path)}_{index}",  # Generate a unique ID
                    'timestamp': pd.to_datetime(raw_alert_details.get(
                        'Timestamp')).timestamp() if 'Timestamp' in raw_alert_details else pd.Timestamp.now().timestamp(),
                    'source_type': 'cic_ids2017',
                    'raw_type': raw_alert_details.get('Label', 'Unknown'),  # Original label
                    'message': message,  # A constructed message
                    # Pass key network details directly if your standardizer uses them
                    'source_ip': raw_alert_details.get('Source IP'),
                    'dest_ip': raw_alert_details.get('Destination IP'),
                    'source_port': raw_alert_details.get('Source Port'),
                    'dest_port': raw_alert_details.get('Destination Port'),
                    'protocol': raw_alert_details.get('Protocol'),  # Protocol is often numeric (6 for TCP, 17 for UDP)
                    # Include other engineered features or all features in a sub-dict
                    'cic_features': {k: v for k, v in raw_alert_details.items() if pd.notna(v)}
                }

                # Use the general AlertPreprocessor for final standardization if desired,
                # or implement full standardization here.
                # For this example, we'll call the general standardizer.
                # It's important that `standardize_alert` can handle these fields.
                standardized_alert = self.alert_standardizer.standardize_alert(raw_alert_for_a2ir,
                                                                               source_type='cic_ids2017')

                if standardized_alert:
                    # Ensure the A²IR type is set correctly based on the label
                    standardized_alert['type'] = a2ir_alert_type
                    standardized_alert['ground_truth_label'] = raw_alert_details.get('Label',
                                                                                     '').strip()  # Keep original label for evaluation
                    standardized_alerts.append(standardized_alert)
                else:
                    logger.warning(f"Failed to standardize CIC-IDS2017 row {index} from {file_path}")

            except Exception as e:
                logger.error(f"Error processing row {index} from {file_path}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully processed and standardized {len(standardized_alerts)} alerts from {file_path}")
        return standardized_alerts

    def process_dataset_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Loads all CIC-IDS2017 CSV files from a directory, processes them, and standardizes alerts.
        """
        all_standardized_alerts = []
        if not os.path.isdir(directory_path):
            logger.error(f"Provided path is not a directory: {directory_path}")
            return []

        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                all_standardized_alerts.extend(self.load_and_process_cic_ids(file_path))

        logger.info(f"Total standardized alerts from directory {directory_path}: {len(all_standardized_alerts)}")
        return all_standardized_alerts