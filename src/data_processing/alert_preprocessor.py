# src/data_processing/alert_preprocessor.py

import json
import csv
import os
import re
import logging
import pandas as pd
from typing import Dict, List, Any, Union, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AlertPreprocessor:
    """
    Preprocesses security alerts from various sources into a standardized format for A²IR.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the alert preprocessor.
        Args:
            config: Configuration dictionary for preprocessing settings.
        """
        self.config = config or {}

        # Regular expressions for extracting information
        self.ip_regex = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        self.email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.url_regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        # More specific hash regex to avoid matching general hex strings
        self.hash_regex = r'\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b'  # MD5, SHA1, SHA256

        # Alert type mapping (can be expanded in config)
        # This mapping is primarily for sources where type isn't explicitly given by a specialized preprocessor
        self.alert_type_mapping = self.config.get('alert_type_mapping', {
            'malicious_email': 'phishing',  # Example from Splunk Attack Range
            'phishing': 'phishing',
            'malware': 'malware',
            'suspicious_download': 'malware',
            'ransomware': 'malware',
            'data_exfiltration': 'insider_threat',
            'lateral_movement': 'insider_threat',
            'brute_force': 'unauthorized_access',  # Can be refined (e.g., ssh_brute_force, ftp_brute_force)
            'denial_of_service': 'ddos',
            'portscan': 'reconnaissance',
            'web attack.*brute force': 'web_attack_brute_force',  # Regex for flexibility
            'web attack.*xss': 'web_attack_xss',
            'web attack.*sql injection': 'web_attack_sqli',
            'infiltration': 'infiltration',
            'bot': 'botnet_activity'
        })
        # Compile regex for alert type mapping if keys are regex
        self.compiled_type_mapping = []
        for pattern, alert_type in self.alert_type_mapping.items():
            try:
                self.compiled_type_mapping.append((re.compile(pattern, re.IGNORECASE), alert_type))
            except re.error:  # If pattern is not a valid regex, use exact match
                self.compiled_type_mapping.append((pattern.lower(), alert_type))

        logger.info("Alert preprocessor initialized")

    def preprocess_file(self, file_path: str, source_type_override: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Preprocess alerts from a file based on its format.
        Args:
            file_path: Path to the file containing alerts
            source_type_override: Optionally override the detected source type
        Returns:
            List of standardized alert dictionaries
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []

        # Determine source type (e.g., 'splunk_attack_range', 'generic_json', 'generic_csv', 'text')
        # This can be inferred from file name patterns or a config, or passed as override
        filename = os.path.basename(file_path).lower()
        effective_source_type = source_type_override
        if not effective_source_type:
            if "splunk_attack_range" in filename or "sar_" in filename:
                effective_source_type = "splunk_attack_range"
            elif "lanl" in filename:
                effective_source_type = "lanl"  # Example, needs specific handling logic
            # Add other source type inferences here
            else:
                effective_source_type = "unknown_file_source"

        file_extension = os.path.splitext(file_path)[1].lower()
        processed_alerts = []
        try:
            if file_extension == '.json':
                processed_alerts = self._process_json_file(file_path, effective_source_type)
            elif file_extension == '.csv':
                # For CSV, we assume it might be generic, unless it's specifically CIC-IDS (handled by its own preprocessor)
                # The generic CSV processing should map columns based on a configuration if possible.
                processed_alerts = self._process_csv_file(file_path, effective_source_type)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  # Added encoding
                    first_line = f.readline().strip()
                    if first_line.startswith('{') and first_line.endswith('}'):
                        processed_alerts = self._process_jsonl_file(file_path, effective_source_type)
                    else:
                        processed_alerts = self._process_text_file(file_path, effective_source_type)
            else:
                logger.error(f"Unsupported file format: {file_extension} for file {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error preprocessing file {file_path}: {e}", exc_info=True)
            return []

        logger.info(f"Standardized {len(processed_alerts)} alerts from {file_path} (source: {effective_source_type})")
        return processed_alerts

    def _process_json_file(self, file_path: str, source_type: str) -> List[Dict[str, Any]]:
        """Process a JSON file containing alerts."""
        alerts_data = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            return []

        if isinstance(data, list):
            alerts_data = data
        elif isinstance(data, dict) and 'alerts' in data and isinstance(data['alerts'], list):
            alerts_data = data['alerts']
        elif isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):  # Splunk format
            alerts_data = data['results']
        elif isinstance(data, dict):  # Single alert
            alerts_data = [data]
        else:
            logger.warning(
                f"Unexpected JSON structure in {file_path}. Expected list of alerts or dict with 'alerts'/'results' key.")

        return [sa for alert in alerts_data if (sa := self.standardize_alert(alert, source_type)) is not None]

    def _process_jsonl_file(self, file_path: str, source_type: str) -> List[Dict[str, Any]]:
        """Process a JSON Lines file (one JSON object per line)."""
        standardized_alerts = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    alert_data = json.loads(line)
                    sa = self.standardize_alert(alert_data, source_type, line_number=i + 1)
                    if sa: standardized_alerts.append(sa)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON on line {i + 1} in {file_path}: {line[:100]}...")
        return standardized_alerts

    def _process_csv_file(self, file_path: str, source_type: str) -> List[Dict[str, Any]]:
        """Process a generic CSV file containing alerts. CIC-IDS2017 should be handled by its own preprocessor."""
        # This is for generic CSVs, not CIC-IDS2017 (which has its dedicated CICIDSPreprocessor)
        if source_type == 'cic_ids2017':
            logger.error(
                "CICIDSPreprocessor should handle CIC-IDS2017 CSVs, not the generic AlertPreprocessor._process_csv_file.")
            return []
        standardized_alerts = []
        try:
            # Attempt to sniff delimiter, or use comma as default
            # For more complex CSVs, might need to allow config for delimiter, quotechar etc.
            df = pd.read_csv(file_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            raw_alerts = df.to_dict('records')
            for i, alert_data in enumerate(raw_alerts):
                sa = self.standardize_alert(alert_data, source_type,
                                            line_number=i + 1)  # line_number for CSVs is row index + 1
                if sa: standardized_alerts.append(sa)
        except Exception as e:
            logger.error(f"Error processing generic CSV file {file_path}: {e}", exc_info=True)
        return standardized_alerts

    def _process_text_file(self, file_path: str, source_type: str) -> List[Dict[str, Any]]:
        """Process a plain text file, assuming one alert per line or a parsable log format."""
        standardized_alerts = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                raw_alert = {'message': line, '_raw': line, 'line_number': i + 1}
                # Potentially add more sophisticated log parsing here if source_type indicates a specific log format
                sa = self.standardize_alert(raw_alert, source_type)
                if sa: standardized_alerts.append(sa)
        return standardized_alerts

    def standardize_alert(self, raw_alert: Dict[str, Any], source_type: str, line_number: Optional[int] = None) -> \
    Optional[Dict[str, Any]]:
        """
        Convert a raw alert into A²IR's standardized format.
        Args:
            raw_alert: Raw alert dictionary from various sources (including specialized preprocessors).
            source_type: Source of the alert (e.g., 'splunk_attack_range', 'cic_ids2017', 'generic_text').
            line_number: Optional line number (for file-based sources).
        Returns:
            Standardized alert dictionary or None if processing fails.
        """
        try:
            # --- Core Information ---
            alert_id = self._extract_id(raw_alert, source_type, line_number)
            timestamp = self._extract_timestamp(raw_alert)
            raw_type_extracted = self._extract_raw_type(raw_alert)  # Original type from source

            # --- Main Data Payload ---
            # Initialize `data_payload` which will become `standard_alert['data']`
            data_payload = {}

            # Message: prioritize direct message, then construct if needed
            if 'message' in raw_alert and raw_alert['message']:
                data_payload['message'] = str(raw_alert['message'])
            else:
                data_payload['message'] = self._construct_message_from_fields(raw_alert)

            # Specific handling for CIC-IDS2017 (features already extracted by CICIDSPreprocessor)
            if source_type == 'cic_ids2017':
                # CICIDSPreprocessor should have put all its features into raw_alert['cic_features']
                # and potentially top-level keys like 'source_ip', 'dest_ip', etc.
                data_payload['cic_features'] = raw_alert.get('cic_features', {})  # Store all original features
                data_payload['source_ip'] = raw_alert.get('Source IP',
                                                          raw_alert.get('source_ip'))  # Prefer specific CIC name
                data_payload['dest_ip'] = raw_alert.get('Destination IP', raw_alert.get('dest_ip'))
                data_payload['source_port'] = raw_alert.get('Source Port', raw_alert.get('source_port'))
                data_payload['dest_port'] = raw_alert.get('Destination Port', raw_alert.get('dest_port'))
                data_payload['protocol'] = raw_alert.get('Protocol', raw_alert.get('protocol'))
            else:  # Generic extraction for other sources
                data_payload['source_ip'] = self._extract_source_ip(raw_alert)
                data_payload['dest_ip'] = self._extract_dest_ip(raw_alert)
                data_payload['username'] = self._extract_username(raw_alert)
                data_payload['protocol'] = self._extract_protocol(raw_alert)

            # Remove None values from data_payload for cleaner output
            data_payload = {k: v for k, v in data_payload.items() if v is not None}

            # --- A²IR Standardized Alert Type ---
            # If raw_alert (from a specialized preprocessor like CICIDS) already has 'type', use it.
            # Otherwise, determine it.
            a2ir_alert_type = raw_alert.get('type')  # Check if specialized preprocessor set this
            if not a2ir_alert_type:
                a2ir_alert_type = self._determine_a2ir_type(raw_alert, raw_type_extracted,
                                                            data_payload.get('message', ''))

            # --- Populate specific data structures based on A²IR type ---
            if a2ir_alert_type == 'phishing' and source_type != 'cic_ids2017':
                email_data = self._extract_email_data(raw_alert, data_payload.get('message', ''))
                if email_data: data_payload['email'] = email_data
            elif a2ir_alert_type == 'malware' and source_type != 'cic_ids2017':
                malware_data = self._extract_malware_data(raw_alert, data_payload.get('message', ''))
                if malware_data: data_payload['malware'] = malware_data
            # Network data for non-CICIDS sources or if cic_features is empty
            elif (a2ir_alert_type == 'ddos' or a2ir_alert_type == 'reconnaissance') and not data_payload.get(
                    'cic_features'):
                network_data = self._extract_network_data_generic(raw_alert, data_payload.get('message', ''))
                if network_data: data_payload['network'] = network_data

            mitre_data = self._extract_mitre_data(raw_alert, data_payload.get('message', ''))
            if mitre_data: data_payload['mitre'] = mitre_data

            standard_alert = {
                'id': alert_id,
                'timestamp': timestamp,
                'source_type': source_type,
                'raw_type': raw_type_extracted,  # The original type from the source log/alert
                'type': a2ir_alert_type,  # A²IR's standardized type
                'data': data_payload,
                'ground_truth_label': raw_alert.get('Label', raw_alert.get('ground_truth_label'))
                # Preserve for evaluation
            }
            standard_alert = {k: v for k, v in standard_alert.items() if v is not None}  # Clean top-level Nones

            return standard_alert

        except Exception as e:
            logger.error(f"Error standardizing alert (ID attempt: {raw_alert.get('id', 'N/A')}): {e}", exc_info=True)
            return None

    def _extract_id(self, raw_alert: Dict[str, Any], source_type: str, line_number: Optional[int] = None) -> str:
        common_id_fields = ['id', '_id', 'alert_id', 'event_id', 'sid', 'uid', 'Flow ID']
        for id_field in common_id_fields:
            if id_field in raw_alert and raw_alert[id_field]:
                return str(raw_alert[id_field])

        # Fallback ID generation
        ts_part = str(self._extract_timestamp(raw_alert)).replace('.', '')
        content_hash_part = str(hash(json.dumps(raw_alert, sort_keys=True)))[-6:]
        ln_part = f"_L{line_number}" if line_number is not None else ""
        return f"{source_type}_{ts_part}_{content_hash_part}{ln_part}"

    def _extract_timestamp(self, raw_alert: Dict[str, Any]) -> float:
        common_ts_fields = ['_time', 'timestamp', 'Timestamp', 'time', 'event_time', 'created_at', 'startTime']
        for ts_field in common_ts_fields:
            if ts_field in raw_alert and raw_alert[ts_field]:
                ts_value = raw_alert[ts_field]
                try:
                    if isinstance(ts_value, (int, float)):
                        return ts_value / 1000 if ts_value > 1e12 else float(ts_value)  # Handle ms
                    if isinstance(ts_value, str):
                        # Try common string formats, including CIC-IDS2017's "DD/MM/YYYY HH:MM:SS"
                        # and ISO8601 variations
                        common_formats = [
                            "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                            "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                            "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S,%f",
                            "%a %b %d %H:%M:%S %Y",  # E.g. 'Mon Jul 03 09:20:06 2017'
                            "%Y-%m-%d %H:%M:%S %Z"
                        ]
                        # Handle timezone-aware strings like "2023-10-26T10:30:00+00:00" or "2023-10-26T10:30:00Z"
                        if 'Z' in ts_value: ts_value = ts_value.replace('Z', '+0000')
                        # Python's %z can be tricky with non-padded offsets, try manual parsing for simple cases
                        if '+' in ts_value[10:] or '-' in ts_value[10:]:  # Check for offset character beyond date part
                            if ts_value[-3] == ':':  # e.g. +00:00
                                ts_value = ts_value[:-3] + ts_value[-2:]

                        for fmt in common_formats:
                            try:
                                return datetime.strptime(ts_value, fmt).timestamp()
                            except ValueError:
                                continue
                        logger.warning(f"Could not parse timestamp string: {ts_value} with common formats.")
                except Exception as e:
                    logger.debug(f"Timestamp parsing error for field {ts_field} value {ts_value}: {e}")
                    continue
        return datetime.now().timestamp()  # Fallback

    def _construct_message_from_fields(self, raw_alert: Dict[str, Any]) -> str:
        # If no primary message field, construct one from other key fields
        # Exclude very long fields or complex structures to keep it readable
        excluded_keys = ['_raw', 'raw', 'data', 'payload', 'cic_features', 'message']  # Exclude already tried 'message'
        message_parts = []
        for key, value in raw_alert.items():
            if key not in excluded_keys:
                if isinstance(value, (str, int, float, bool)):
                    if len(str(value)) < 200:  # Limit length of individual field values
                        message_parts.append(f"{key}: {value}")
                elif isinstance(value, list) and len(value) < 10:  # Limit list length
                    try:
                        str_val = json.dumps(value)
                        if len(str_val) < 200: message_parts.append(f"{key}: {str_val}")
                    except TypeError:
                        pass  # Skip if not JSON serializable
        return "; ".join(message_parts) if message_parts else "No message content available"

    def _extract_message(self, raw_alert: Dict[str, Any]) -> str:  # Kept for internal use by other extractors
        """Extract the alert message. (Internal use by other extractors)"""
        if 'message' in raw_alert and raw_alert['message']: return str(raw_alert['message'])
        for msg_field in ['msg', '_raw', 'description', 'summary', 'event_message', 'title', 'details']:
            if msg_field in raw_alert and raw_alert[msg_field]:
                return str(raw_alert[msg_field])
        return self._construct_message_from_fields(raw_alert)

    def _extract_raw_type(self, raw_alert: Dict[str, Any]) -> str:
        common_type_fields = ['type', 'raw_type', 'alert_type', 'event_type', 'category', 'signature', 'rule_name',
                              'name', 'event_name', 'Label']
        for type_field in common_type_fields:
            if type_field in raw_alert and raw_alert[type_field]:
                return str(raw_alert[type_field]).strip()
        return "unknown_raw_type"

    def _extract_source_ip(self, raw_alert: Dict[str, Any]) -> Optional[str]:
        common_fields = ['src_ip', 'source_ip', 'Source IP', 'src', 'sourceAddress', 'src_addr', 'client_ip']
        for ip_field in common_fields:
            if ip_field in raw_alert and self._is_valid_ip(raw_alert[ip_field]):
                return str(raw_alert[ip_field]).strip()
        message = self._extract_message(raw_alert)
        src_match = re.search(r'(?:src|source|from|client)[\s:=]+(\d+\.\d+\.\d+\.\d+)', message, re.IGNORECASE)
        if src_match: return src_match.group(1)
        ip_matches = re.findall(self.ip_regex, message)
        return ip_matches[0] if ip_matches else None

    def _extract_dest_ip(self, raw_alert: Dict[str, Any]) -> Optional[str]:
        common_fields = ['dst_ip', 'dest_ip', 'Destination IP', 'dst', 'destinationAddress', 'dest_addr', 'server_ip']
        for ip_field in common_fields:
            if ip_field in raw_alert and self._is_valid_ip(raw_alert[ip_field]):
                return str(raw_alert[ip_field]).strip()
        message = self._extract_message(raw_alert)
        dst_match = re.search(r'(?:dst|destination|to|server)[\s:=]+(\d+\.\d+\.\d+\.\d+)', message, re.IGNORECASE)
        if dst_match: return dst_match.group(1)
        ip_matches = re.findall(self.ip_regex, message)
        src_ip = self._extract_source_ip(raw_alert)  # Avoid returning src_ip as dst_ip
        for ip_match in ip_matches:
            if ip_match != src_ip: return ip_match
        return ip_matches[0] if ip_matches and ip_matches[0] != src_ip else (
            ip_matches[1] if len(ip_matches) > 1 and ip_matches[1] != src_ip else None)

    def _extract_username(self, raw_alert: Dict[str, Any]) -> Optional[str]:
        common_fields = ['username', 'user', 'user_name', 'userId', 'account', 'actor']
        for user_field in common_fields:
            if user_field in raw_alert and raw_alert[user_field]:
                return str(raw_alert[user_field])
        message = self._extract_message(raw_alert)
        user_match = re.search(r'(?:user|username|account|actor)[\s:=]+(["\']?)([a-zA-Z0-9._-]+)\1', message,
                               re.IGNORECASE)
        return user_match.group(2) if user_match else None

    def _extract_protocol(self, raw_alert: Dict[str, Any]) -> Optional[str]:
        common_fields = ['protocol', 'Protocol', 'proto', 'transport']
        for proto_field in common_fields:
            if proto_field in raw_alert and raw_alert[proto_field]:
                val = str(raw_alert[proto_field]).upper()
                # Handle numeric protocols common in CIC-IDS
                if val == '6': return 'TCP'
                if val == '17': return 'UDP'
                if val == '1': return 'ICMP'
                return val if val.isalpha() else None  # Basic check
        message = self._extract_message(raw_alert)
        proto_match = re.search(r'(?:protocol|proto)[\s:=]+(\w+)', message, re.IGNORECASE)
        if proto_match: return proto_match.group(1).upper()
        for proto in ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'SMTP', 'FTP', 'SSH']:
            if re.search(rf'\b{proto}\b', message, re.IGNORECASE): return proto.upper()
        return None

    def _determine_a2ir_type(self, raw_alert: Dict[str, Any], raw_type: str, message: str) -> str:
        """Determine the standardized A²IR alert type."""
        # Try direct mapping from raw_type first
        raw_type_lower = raw_type.lower()
        for pattern, alert_type in self.compiled_type_mapping:
            if isinstance(pattern, re.Pattern):  # Compiled regex
                if pattern.search(raw_type_lower): return alert_type
            elif isinstance(pattern, str):  # Exact string match (already lowercased)
                if pattern == raw_type_lower: return alert_type

        # If no mapping found, try to determine from message content
        message_lower = message.lower()
        # Order matters: more specific terms first
        if any(term in message_lower for term in
               ['phish', 'spoofed email', 'credential harvest', 'malicious mail']): return 'phishing'
        if any(term in message_lower for term in
               ['malware', 'virus', 'trojan', 'ransomware', 'worm', 'backdoor', 'rootkit', 'keylogger', 'spyware',
                'adware', 'coinminer', 'botnet command', 'c2 communication']): return 'malware'
        if any(term in message_lower for term in
               ['ddos', 'distributed denial of service', 'dos', 'denial of service', 'udp flood', 'syn flood',
                'icmp flood', 'http flood', 'traffic spike', 'bandwidth saturation',
                'service unavailable']): return 'ddos'
        if any(term in message_lower for term in ['port scan', 'network scan', 'host discovery', 'service scan', 'nmap',
                                                  'masscan']): return 'reconnaissance'
        if any(term in message_lower for term in
               ['brute force', 'failed login attempts', 'password guessing', 'ssh-patator',
                'ftp-patator']): return 'brute_force'  # Could be web_brute_force or other
        if any(term in message_lower for term in
               ['web attack', 'xss', 'cross-site scripting', 'sql injection', 'sqli', 'directory traversal',
                'command injection']): return 'web_attack'  # Generic, can be refined
        if any(term in message_lower for term in
               ['insider threat', 'data theft', 'data exfiltration', 'unauthorized access',
                'privilege escalation']): return 'insider_threat'
        if any(term in message_lower for term in
               ['exploit', 'vulnerability', 'zeroday', '0-day', 'heartbleed']): return 'exploit'
        if any(term in message_lower for term in
               ['unauthorized connection', 'policy violation']): return 'policy_violation'

        # Fallback based on keywords from compiled_type_mapping applied to message
        for pattern, alert_type in self.compiled_type_mapping:
            if isinstance(pattern, re.Pattern):
                if pattern.search(message_lower): return alert_type
            elif isinstance(pattern, str):
                if pattern in message_lower: return alert_type  # Substring match for non-regex patterns

        return 'unknown_event_type'  # More descriptive default

    def _extract_email_data(self, raw_alert: Dict[str, Any], message: str) -> Optional[Dict[str, Any]]:
        email_data = {}
        for field_map in [{'key': 'sender', 'alt_keys': ['from', 'sender_address', 'src_email', 'mail_from']},
                          {'key': 'subject', 'alt_keys': ['email_subject', 'mail_subject']},
                          {'key': 'recipients', 'alt_keys': ['to', 'rcpt_to', 'mail_to']}]:
            for key_attempt in [field_map['key']] + field_map['alt_keys']:
                if key_attempt in raw_alert and raw_alert[key_attempt]:
                    email_data[field_map['key']] = raw_alert[key_attempt]
                    break

        if 'sender' not in email_data:
            email_matches = re.findall(self.email_regex, message)
            if email_matches: email_data['sender'] = email_matches[0]  # Simplistic: assumes first is sender

        if 'subject' not in email_data:
            subject_match = re.search(r'subject[\s:=]+(["\']?)(.*?)\1(?:[\r\n]|$)', message, re.IGNORECASE | re.DOTALL)
            if subject_match: email_data['subject'] = subject_match.group(2).strip()

        # URLs (more robust extraction)
        urls_found = set()
        if 'urls' in raw_alert and isinstance(raw_alert['urls'], list): urls_found.update(raw_alert['urls'])
        if 'url' in raw_alert and isinstance(raw_alert['url'], str): urls_found.add(raw_alert['url'])
        urls_in_message = re.findall(self.url_regex, message)
        urls_found.update(urls_in_message)
        if urls_found: email_data['urls'] = list(urls_found)

        # Attachments
        if 'attachments' in raw_alert:
            attachments = raw_alert['attachments']
            if isinstance(attachments, list):
                email_data['attachments'] = attachments
            elif isinstance(attachments, str):
                email_data['attachments'] = [att.strip() for att in attachments.split(',') if att.strip()]
        elif 'attachment_name' in raw_alert:
            email_data['attachments'] = [raw_alert['attachment_name']]

        return email_data if email_data else None

    def _extract_malware_data(self, raw_alert: Dict[str, Any], message: str) -> Optional[Dict[str, Any]]:
        malware_data = {}
        hashes_found = set()
        for field in ['hash', 'file_hash', 'md5', 'sha1', 'sha256', 'FileHash', 'ProcessHash']:
            if field in raw_alert and raw_alert[field]:
                if isinstance(raw_alert[field], list):
                    for h in raw_alert[field]:
                        if isinstance(h, str) and re.fullmatch(self.hash_regex, h): hashes_found.add(h)
                elif isinstance(raw_alert[field], str) and re.fullmatch(self.hash_regex, raw_alert[field]):
                    hashes_found.add(raw_alert[field])
        hashes_in_message = re.findall(self.hash_regex, message)
        hashes_found.update(hashes_in_message)
        if hashes_found: malware_data['hashes'] = list(hashes_found)

        for field in ['malware_name', 'malware_family', 'virus_name', 'threat_name', 'signature_name']:
            if field in raw_alert and raw_alert[field]: malware_data['name'] = str(raw_alert[field]); break
        for field in ['file_name', 'filename', 'FileName', 'target_file', 'image_path', 'ProcessName', 'ImagePath']:
            if field in raw_alert and raw_alert[field]: malware_data['file_path'] = str(
                raw_alert[field]); break  # Changed to file_path

        processes = []
        if 'process_name' in raw_alert: processes.append(raw_alert['process_name'])
        if 'process_path' in raw_alert: processes.append(raw_alert['process_path'])
        # Crude process extraction from message
        proc_match = re.search(
            r'(?:process|image|application)[\s:=]+(["\']?)([\w.\\/:-]+\.(?:exe|dll|scr|bat|sh|py))\1', message,
            re.IGNORECASE)
        if proc_match: processes.append(proc_match.group(2))
        if processes: malware_data['processes'] = list(set(processes))

        return malware_data if malware_data else None

    def _extract_network_data_generic(self, raw_alert: Dict[str, Any], message: str) -> Optional[Dict[str, Any]]:
        """Extract generic network-specific data if not CIC-IDS2017."""
        # This is a fallback for non-CICIDS data or when cic_features are missing
        network_data = {}
        for field_map in [
            ('packet_count', ['packet_count', 'packets', 'packet_total', 'Total Packets']),
            ('byte_count', ['byte_count', 'bytes', 'byte_total', 'Total Bytes']),
            ('duration', ['duration', 'flow_duration', 'Flow Duration'])
        ]:
            key, alt_keys = field_map
            for attempt_key in alt_keys:
                if attempt_key in raw_alert and raw_alert[attempt_key]:
                    try:
                        network_data[key] = float(raw_alert[attempt_key]); break
                    except (ValueError, TypeError):
                        pass

        ports = set()
        for port_field in ['port', 'dest_port', 'dst_port', 'Destination Port', 'src_port', 'source_port',
                           'Source Port', 'ports']:
            if port_field in raw_alert and raw_alert[port_field]:
                val = raw_alert[port_field]
                try:
                    if isinstance(val, list):
                        for p_item in val: ports.add(int(p_item))
                    elif isinstance(val, str) and ',' in val:  # Comma-separated
                        for p_str in val.split(','): ports.add(int(p_str.strip()))
                    else:
                        ports.add(int(val))
                except (ValueError, TypeError):
                    pass
        if ports: network_data['ports'] = [p for p in list(ports) if 0 <= p <= 65535]

        return network_data if network_data else None

    def _extract_mitre_data(self, raw_alert: Dict[str, Any], message: str) -> Optional[Dict[str, Any]]:
        mitre_data = {}
        for field_map in [
            ('tactic', ['mitre_tactic', 'tactic', 'attack_tactic', 'MITRE ATT&CK Tactic']),
            ('technique', ['mitre_technique', 'technique', 'attack_technique', 'MITRE ATT&CK Technique']),
            ('technique_id', ['mitre_technique_id', 'technique_id', 'MITRE ID'])
        ]:
            key, alt_keys = field_map
            for attempt_key in alt_keys:
                if attempt_key in raw_alert and raw_alert[attempt_key]:
                    mitre_data[key] = str(raw_alert[attempt_key]);
                    break

        # Regex for Txxxx or Txxxx.xxx format
        mitre_id_match = re.search(r'(T\d{4}(?:\.\d{3})?)', message)
        if mitre_id_match and 'technique_id' not in mitre_data:
            mitre_data['technique_id'] = mitre_id_match.group(1)

        return mitre_data if mitre_data else None

    def _is_valid_ip(self, ip_str: Optional[Any]) -> bool:
        if not isinstance(ip_str, str): return False
        ip_str = ip_str.strip()
        # Basic regex for IPv4, doesn't validate ranges perfectly but good enough for common cases
        if re.fullmatch(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$", ip_str):
            parts = ip_str.split('.')
            try:
                return all(0 <= int(part) <= 255 for part in parts)
            except ValueError:
                return False
        # Add IPv6 check if needed:
        # elif re.fullmatch(r"([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|...", ip_str):
        #    return True
        return False


# Standalone function for orchestrator to call (optional, can be removed if orchestrator uses class methods)
def run_preprocessing(config: Dict[str, Any], input_file: str = None, output_file: str = None,
                      source_type_override: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run preprocessing on specified input data.
    Args:
        config: Application configuration.
        input_file: Specific input file to process.
        output_file: Optional output file to save processed alerts.
        source_type_override: Optional string to override source type detection.
    Returns:
        List of processed (standardized) alerts.
    """
    preprocessor = AlertPreprocessor(config.get('preprocessing', {}))
    all_processed_alerts = []

    if input_file:  # Process a single specified file
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return []
        all_processed_alerts = preprocessor.preprocess_file(input_file, source_type_override)
    else:  # Process datasets defined in config if no specific input_file
        datasets_config = config.get('datasets', {})
        if not datasets_config:
            logger.warning("No input file specified and no datasets found in configuration for generic preprocessing.")
            return []

        logger.info("Processing datasets specified in configuration for generic preprocessing...")
        for dataset_name, dataset_cfg in datasets_config.items():
            dataset_path = dataset_cfg.get('path')
            if not dataset_path or not os.path.exists(dataset_path):
                logger.warning(f"Path not found or not specified for dataset '{dataset_name}'. Skipping.")
                continue

            # If dataset_path is a directory, process all valid files within it
            if os.path.isdir(dataset_path):
                for filename in os.listdir(dataset_path):
                    # Define what constitutes a "valid" file, e.g., by extension
                    # This is a generic handler, specific preprocessors like CICIDS handle their own directory logic
                    if filename.lower().endswith(('.json', '.csv', '.txt', '.log')):
                        file_to_process = os.path.join(dataset_path, filename)
                        logger.info(f"Preprocessing file from dataset '{dataset_name}': {file_to_process}")
                        all_processed_alerts.extend(preprocessor.preprocess_file(file_to_process,
                                                                                 dataset_name))  # Use dataset_name as source_type
            elif os.path.isfile(dataset_path):  # If it's a single file
                logger.info(f"Preprocessing dataset file '{dataset_name}': {dataset_path}")
                all_processed_alerts.extend(preprocessor.preprocess_file(dataset_path, dataset_name))

    if output_file and all_processed_alerts:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_processed_alerts, f, indent=2)
            logger.info(f"Saved {len(all_processed_alerts)} processed alerts to {output_file}")
        except Exception as e:
            logger.error(f"Error saving processed alerts to {output_file}: {e}")

    logger.info(f"Preprocessing complete. Total alerts standardized by AlertPreprocessor: {len(all_processed_alerts)}")
    return all_processed_alerts