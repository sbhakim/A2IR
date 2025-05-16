# tests/unit/test_cic_ids_preprocessor.py
import pytest
import pandas as pd
import os
from typing import Dict, List
from src.data_processing.dataset_preprocessors.cic_ids_preprocessor import CICIDSPreprocessor
from src.data_processing.alert_preprocessor import AlertPreprocessor  # For instantiation


# Mock AlertPreprocessor for testing CICIDSPreprocessor in isolation if needed,
# or use the real one for integration-style unit testing.
class MockAlertStandardizer:
    def standardize_alert(self, raw_alert: Dict, source_type: str) -> Dict:
        # Simple mock: just return a dict with type and id
        return {"id": raw_alert.get("id", "mock_id"), "type": "mock_network_event", "data": raw_alert}


@pytest.fixture
def cic_ids_sample_data_path(tmp_path):
    # Create a dummy CIC-IDS2017 CSV file for testing
    data = {
        'Flow ID': ['1.0.0.1-1.0.0.2-12345-54321-6'],
        ' Source IP': ['1.0.0.1 '],  # Note leading/trailing spaces
        ' Source Port': [12345],
        ' Destination IP': [' 1.0.0.2'],
        ' Destination Port': [54321],
        ' Protocol': [6],
        ' Timestamp': ['20/02/2017 09:30:00'],  # Example timestamp format
        ' Flow Duration': [1000],
        ' Total Fwd Packets': [2],
        ' Total Backward Packets': [1],
        'Fwd PSH Flags': [0],
        'SYN Flag Count': [1],
        'ACK Flag Count': [1],
        'URG Flag Count': [0],
        ' Packet Length Mean': [50.5],
        ' Average Packet Size': [60.0],
        ' Subflow Fwd Bytes': [100],
        ' Subflow Bwd Bytes': [50],
        ' Init_Win_bytes_forward': [8192],
        ' Init_Win_bytes_backward': [8192],
        ' Active Mean': [100.0],
        ' Idle Mean': [500.0],
        ' Label': [' DDoS ']  # Note leading/trailing spaces
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_cicids.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_cic_ids_preprocessor_load_single_file(cic_ids_sample_data_path):
    # alert_standardizer = MockAlertStandardizer() # Use mock
    alert_standardizer = AlertPreprocessor()  # Or use real for more integrated test
    preprocessor = CICIDSPreprocessor(alert_standardizer=alert_standardizer)

    processed_alerts = preprocessor.load_and_process_cic_ids(str(cic_ids_sample_data_path))

    assert len(processed_alerts) == 1
    alert = processed_alerts[0]
    assert alert['type'] == 'ddos'  # Based on LABEL_TO_A2IR_TYPE
    assert 'cic_features' in alert['data']  # From our new structure in standardize_alert
    assert alert['data']['source_ip'] == '1.0.0.1'  # Check if spaces were stripped by your logic
    assert alert['data']['cic_features']['Destination IP'] == ' 1.0.0.2'  # Original value with space
    assert alert['ground_truth_label'] == 'DDoS'


def test_cic_ids_preprocessor_directory(tmp_path):
    alert_standardizer = AlertPreprocessor()
    preprocessor = CICIDSPreprocessor(alert_standardizer=alert_standardizer)

    # Create two dummy files in the directory
    data1 = {'Source IP': ['1.1.1.1'], 'Destination IP': ['2.2.2.2'], 'Protocol': [6], 'Label': ['BENIGN'],
             'Timestamp': ['20/02/2017 09:30:00']}
    data2 = {'Source IP': ['3.3.3.3'], 'Destination IP': ['4.4.4.4'], 'Protocol': [17], 'Label': ['PortScan'],
             'Timestamp': ['20/02/2017 09:31:00']}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    file1_path = tmp_path / "cic_day1.csv"
    file2_path = tmp_path / "cic_day2.csv"
    df1.to_csv(file1_path, index=False)
    df2.to_csv(file2_path, index=False)

    # Create a non-csv file that should be ignored
    (tmp_path / "notes.txt").write_text("ignore me")

    processed_alerts = preprocessor.process_dataset_directory(str(tmp_path))
    assert len(processed_alerts) == 2
    assert processed_alerts[0]['type'] == 'benign_traffic'
    assert processed_alerts[1]['type'] == 'reconnaissance'