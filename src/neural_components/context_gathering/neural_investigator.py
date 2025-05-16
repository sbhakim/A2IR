# src/neural_components/context_gathering/neural_investigator.py

import logging
import os
import json
import requests  # Ensure this is installed
import socket
import time
import whois  # Ensure this is installed
import dns.resolver  # Ensure this is installed
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Import the new strategy classes
from .investigation_strategies.ip_investigator_strategy import IPInvestigationStrategy
from .investigation_strategies.url_investigator_strategy import URLInvestigationStrategy
from .investigation_strategies.domain_investigator_strategy import DomainInvestigationStrategy  # Create this
from .investigation_strategies.file_hash_investigator_strategy import FileHashInvestigationStrategy  # Create this
from .investigation_strategies.email_investigator_strategy import EmailInvestigationStrategy  # Create this
from .investigation_strategies.user_investigator_strategy import UserInvestigationStrategy  # Create this

logger = logging.getLogger(__name__)


class NeuralInvestigator:
    """
    Neural component for context gathering in the A²IR framework.
    This implements the Ni component in the neurosymbolic integration pathway:
    A²IR = Nt[S(KG, R, I)(Ni)]

    The investigator gathers additional context about security incidents when
    the initial classification confidence is low or more information is needed.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the neural investigator.
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # API configurations
        self.virustotal_api_key = self.config.get('virustotal_api_key')
        self.shodan_api_key = self.config.get('shodan_api_key')
        self.abuseipdb_api_key = self.config.get('abuseipdb_api_key')

        # Timeouts and limits
        self.timeout = self.config.get('timeout', 10)
        self.max_retries = self.config.get('max_retries', 3)

        # Cache for API results
        self.cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour

        # API helpers (can be a separate class or just methods here)
        self.api_helpers = self  # Or a dedicated ApiHelper class instance

        # Initialize investigation strategies
        # The cache_manager will be 'self' as it has _get_from_cache and _set_cache
        self.ip_strategy = IPInvestigationStrategy(cache_manager=self, api_helpers=self.api_helpers)
        # For URL strategy, it needs domain investigator, so domain needs to be initialized first or passed carefully
        self.domain_strategy = DomainInvestigationStrategy(cache_manager=self,
                                                           api_helpers=self.api_helpers)  # Create this class
        self.url_strategy = URLInvestigationStrategy(cache_manager=self, api_helpers=self.api_helpers,
                                                     domain_investigator=self.domain_strategy)
        self.hash_strategy = FileHashInvestigationStrategy(cache_manager=self,
                                                           api_helpers=self.api_helpers)  # Create this class
        self.email_strategy = EmailInvestigationStrategy(cache_manager=self, api_helpers=self.api_helpers,
                                                         domain_investigator=self.domain_strategy)  # Create this class
        self.user_strategy = UserInvestigationStrategy(
            api_helpers=self.api_helpers)  # Create this class, cache might not be relevant here or handle internally

        logger.info("Neural investigator initialized with strategies")

    def investigate(self,
                    investigation_targets: List[Dict[str, Any]],
                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform investigation based on targets.
        Args:
            investigation_targets: List of targets to investigate
            context: Current context information
        Returns:
            Investigation results with additional context
        """
        results = {
            'features': {},
            'entities': {},
            'context': {},
            'timestamp': datetime.now().isoformat()
        }

        for target in investigation_targets:
            target_type = target.get('type')
            target_value = target.get('value')

            logger.info(f"Investigating {target_type}: {target_value}")
            entity_info = None
            try:
                if target_type == 'ip':
                    entity_info = self.ip_strategy.investigate_ip(target_value)
                elif target_type == 'url':
                    entity_info = self.url_strategy.investigate_url(target_value)
                elif target_type == 'domain':
                    entity_info = self.domain_strategy.investigate_domain(target_value)
                elif target_type == 'hash':
                    entity_info = self.hash_strategy.investigate_file_hash(target_value)
                elif target_type == 'email':
                    entity_info = self.email_strategy.investigate_email(target_value)
                elif target_type == 'user':
                    entity_info = self.user_strategy.investigate_user(target_value, context)
                elif target_type == 'general':
                    general_context = self._gather_general_context(context)
                    results['context'].update(general_context)
                    continue  # Skip entity update for general context

                if entity_info:
                    results['entities'][target_type] = results['entities'].get(target_type, [])
                    results['entities'][target_type].append({
                        'value': target_value,
                        'investigation': entity_info
                    })

            except Exception as e:
                logger.error(f"Error investigating {target_type} {target_value}: {e}", exc_info=True)
                results['entities'][target_type] = results['entities'].get(target_type, [])
                results['entities'][target_type].append({
                    'value': target_value,
                    'error': str(e)
                })

        # Analyze patterns across investigations
        patterns = self._analyze_investigation_patterns(results)
        results['patterns'] = patterns

        # Generate investigation features
        features = self._generate_investigation_features(results, context)
        results['features'].update(features)

        return results

    # --- Keep shared helper methods or move them to a utility/base class ---
    # Methods like _get_ip_geolocation, _check_ip_reputation, etc. are now part of api_helpers
    # or should be defined here if they use self.virustotal_api_key etc. directly.
    # For brevity, I'll assume these are defined (or imported) within their respective strategy files or an ApiHelper class.

    def _get_ip_geolocation(self, ip_address: str) -> Dict[str, Any]:
        """Get geolocation information for an IP address. (Example, real call needed)"""
        logger.debug(f"API CALL (SIMULATED): Geolocation for {ip_address}")
        # Simulate; in production, this would use a real geolocation API like MaxMind, ip-api.com
        # For example, using ip-api.com (requires no key for free tier, but rate limited)
        # try:
        #     response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=self.timeout)
        #     response.raise_for_status()
        #     data = response.json()
        #     if data.get("status") == "success":
        #         return {
        #             'country': data.get('country'),
        #             'city': data.get('city'),
        #             'latitude': data.get('lat'),
        #             'longitude': data.get('lon'),
        #             'organization': data.get('org')
        #         }
        # except requests.RequestException as e:
        #     logger.warning(f"ip-api.com request failed for {ip_address}: {e}")
        return {
            'country': 'Unknown (Simulated)', 'city': 'Unknown', 'latitude': 0.0,
            'longitude': 0.0, 'organization': 'Unknown'
        }

    def _check_ip_reputation(self, ip_address: str) -> Dict[str, Any]:
        """Check IP reputation. (Example, real call needed)"""
        logger.debug(f"API CALL (SIMULATED): IP Reputation for {ip_address}")
        # In production, use AbuseIPDB, VirusTotal IP reports, etc.
        # Example with AbuseIPDB (requires API key)
        # if self.abuseipdb_api_key:
        #     headers = {'Key': self.abuseipdb_api_key, 'Accept': 'application/json'}
        #     params = {'ipAddress': ip_address, 'maxAgeInDays': '90'}
        #     try:
        #         response = requests.get('https://api.abuseipdb.com/api/v2/check',
        #                                 headers=headers, params=params, timeout=self.timeout)
        #         response.raise_for_status()
        #         data = response.json().get('data', {})
        #         return {
        #             'reputation_score': data.get('abuseConfidenceScore'),
        #             'is_malicious': data.get('abuseConfidenceScore', 0) > 75, # Example threshold
        #             'abuse_reports': data.get('totalReports'),
        #             'last_reported': data.get('lastReportedAt')
        #         }
        #     except requests.RequestException as e:
        #         logger.warning(f"AbuseIPDB request failed for {ip_address}: {e}")
        return {
            'reputation_score': 0.5, 'is_malicious': False,
            'abuse_reports': 0, 'last_reported': None
        }

    def _get_network_info(self, ip_address: str) -> Dict[str, Any]:
        """Get network information for an IP. (Example, real call needed)"""
        logger.debug(f"API CALL (SIMULATED): Network Info for {ip_address}")
        # In production, use services like Shodan (requires API key) or RIPEstat
        # if self.shodan_api_key:
        #     try:
        #         api = shodan.Shodan(self.shodan_api_key)
        #         host = api.host(ip_address)
        #         return {
        #             'asn': host.get('asn'),
        #             'organization': host.get('org'),
        #             'network_name': host.get('isp'), # ISP often aligns with network name
        #             'ports': host.get('ports')
        #         }
        #     except shodan.APIError as e:
        #         logger.warning(f"Shodan API error for {ip_address}: {e}")
        return {
            'asn': 'AS00000 (Simulated)', 'organization': 'Unknown Organization',
            'network_name': 'Unknown Network'
        }

    def _check_url_reputation(self, url: str) -> Dict[str, Any]:
        """Check URL reputation. (Example, real call needed)"""
        logger.debug(f"API CALL (SIMULATED): URL Reputation for {url}")
        # In production, use VirusTotal URL scan, Google Safe Browse, etc.
        # if self.virustotal_api_key:
        #     params = {'apikey': self.virustotal_api_key, 'resource': url}
        #     try:
        #         response = requests.get('https://www.virustotal.com/vtapi/v2/url/report',
        #                                 params=params, timeout=self.timeout)
        #         response.raise_for_status()
        #         data = response.json()
        #         if data.get('response_code') == 1:
        #             return {
        #                 'is_malicious': data.get('positives', 0) > 0,
        #                 'category': 'N/A', # VT doesn't provide this directly in basic report
        #                 'risk_score': data.get('positives', 0) / data.get('total', 1) if data.get('total', 0) > 0 else 0,
        #                 'positives': data.get('positives'),
        #                 'total_scans': data.get('total'),
        #                 'scan_date': data.get('scan_date')
        #             }
        #     except requests.RequestException as e:
        #         logger.warning(f"VirusTotal URL report failed for {url}: {e}")

        return {
            'is_malicious': False, 'category': 'uncategorized (Simulated)',
            'risk_score': 0.3
        }

    def _check_url_alive(self, url: str) -> Dict[str, Any]:
        """Check if URL is alive."""
        logger.debug(f"Checking aliveness for URL: {url}")
        try:
            response = requests.head(url, timeout=self.timeout, allow_redirects=True,
                                     verify=False)  # Added verify=False for simplicity, use with caution
            return {
                'is_alive': True,
                'status_code': response.status_code,
                'final_url': response.url
            }
        except requests.exceptions.RequestException as e:  # More specific exception
            logger.warning(f"URL aliveness check failed for {url}: {type(e).__name__}")
            return {
                'is_alive': False,
                'status_code': None,
                'final_url': None,
                'error': type(e).__name__
            }

    def _get_dns_records(self, domain: str) -> Dict[str, Any]:
        """Get DNS records for a domain."""
        # This method is used by DomainInvestigationStrategy and EmailInvestigationStrategy
        # It can remain here or be part of an ApiHelper class.
        records = {}
        record_types = ['A', 'MX', 'TXT', 'NS', 'CNAME']
        logger.debug(f"Fetching DNS records for: {domain}")
        for r_type in record_types:
            try:
                answers = dns.resolver.resolve(domain, r_type)
                records[r_type] = [str(rdata) for rdata in answers]
            except dns.resolver.NoAnswer:
                records[r_type] = ["NoAnswer"]
            except dns.resolver.NXDOMAIN:
                records[r_type] = ["NXDOMAIN"]
                break  # No point checking other records if domain doesn't exist
            except dns.exception.Timeout:
                records[r_type] = ["Timeout"]
            except Exception as e:
                logger.warning(f"DNS lookup for {r_type} failed for {domain}: {e}")
                records[r_type] = [f"Error: {type(e).__name__}"]
        return records

    def _get_ssl_info(self, domain: str) -> Dict[str, Any]:
        """Get SSL certificate information. (Example, real call needed)"""
        # This method is used by DomainInvestigationStrategy.
        # It can remain here or be part of an ApiHelper class.
        logger.debug(f"API CALL (SIMULATED): SSL Info for {domain}")
        # In production, use libraries like 'ssl', 'pyOpenSSL', or 'cryptography'
        # to connect to port 443 and inspect the certificate.
        # import ssl
        # import socket
        # from datetime import datetime
        # context = ssl.create_default_context()
        # try:
        #     with socket.create_connection((domain, 443), timeout=self.timeout) as sock:
        #         with context.wrap_socket(sock, server_hostname=domain) as ssock:
        #             cert = ssock.getpeercert()
        #             return {
        #                 'has_ssl': True,
        #                 'issuer': dict(x[0] for x in cert.get('issuer', [])),
        #                 'subject': dict(x[0] for x in cert.get('subject', [])),
        #                 'valid_from': cert.get('notBefore'),
        #                 'valid_until': cert.get('notAfter'),
        #                 'serial_number': cert.get('serialNumber'),
        #                 'version': cert.get('version'),
        #                 # Check validity (simplified)
        #                 'is_valid': datetime.strptime(cert['notBefore'], '%b %d %H:%M:%S %Y %Z') < datetime.utcnow() < datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
        #             }
        # except (ssl.SSLError, socket.error, TimeoutError) as e:
        #     logger.warning(f"SSL info lookup failed for {domain}: {e}")
        #     return {'has_ssl': False, 'error': str(e)}
        return {
            'has_ssl': True, 'issuer': 'Unknown CA (Simulated)',
            'valid_from': datetime.now().isoformat(),
            'valid_until': (datetime.now() + timedelta(days=365)).isoformat(),
            'is_valid': True
        }

    def _check_virustotal(self, file_hash: str) -> Dict[str, Any]:
        """Check file hash with VirusTotal. (Example, real call needed)"""
        # This method is used by FileHashInvestigationStrategy.
        logger.debug(f"API CALL (SIMULATED): VirusTotal for hash {file_hash}")
        # if self.virustotal_api_key:
        #     url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
        #     headers = {"x-apikey": self.virustotal_api_key}
        #     try:
        #         response = requests.get(url, headers=headers, timeout=self.timeout)
        #         response.raise_for_status()
        #         data = response.json().get("data", {}).get("attributes", {})
        #         last_analysis_stats = data.get("last_analysis_stats", {})
        #         return {
        #             'detection_ratio': f"{last_analysis_stats.get('malicious', 0)}/{sum(last_analysis_stats.values())}",
        #             'scan_date': datetime.utcfromtimestamp(data.get("last_analysis_date", time.time())).isoformat() if data.get("last_analysis_date") else None,
        #             'permalink': f"https://www.virustotal.com/gui/file/{file_hash}",
        #             'positives': last_analysis_stats.get('malicious', 0),
        #             'total_scans': sum(last_analysis_stats.values()),
        #             'results': data.get("last_analysis_results") # This can be large
        #         }
        #     except requests.RequestException as e:
        #         logger.warning(f"VirusTotal API request failed for hash {file_hash}: {e}")
        #         return {'error': str(e), 'details': 'API request failed'}
        #     except KeyError:
        #         logger.warning(f"VirusTotal response malformed for hash {file_hash}")
        #         return {'error': 'Malformed API response'}
        # else:
        #     return {'error': 'VirusTotal API key not configured'}
        return {
            'detection_ratio': '0/70 (Simulated)',
            'scan_date': datetime.now().isoformat(),
            'permalink': f'https://www.virustotal.com/file/{file_hash}'
        }

    # --- Caching Methods ---
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            cached_item = self.cache[key]
            if datetime.now().timestamp() - cached_item['timestamp'] < self.cache_ttl:
                logger.debug(f"Cache hit for key: {key}")
                return cached_item['data']
            else:
                logger.debug(f"Cache expired for key: {key}")
                del self.cache[key]
        logger.debug(f"Cache miss for key: {key}")
        return None

    def _set_cache(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = {
            'data': value,
            'timestamp': datetime.now().timestamp()
        }
        logger.debug(f"Cached value for key: {key}")

    # --- General Context Gathering and Analysis Methods (can remain here) ---
    def _gather_general_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather general contextual information."""
        # (Original _gather_general_context method code from source: 837 to source: 840)
        # This method itself uses other _get_*_context methods.
        # Keep these helpers or make them part of a context utility.
        general_context = {
            'timestamp': datetime.now().isoformat()
        }
        try:
            network_context = self._get_network_context(context)
            general_context['network'] = network_context
        except Exception as e:
            logger.warning(f"Network context gathering failed: {e}")
            general_context['network'] = {'error': str(e)}
        try:
            system_context = self._get_system_context(context)
            general_context['system'] = system_context
        except Exception as e:
            logger.warning(f"System context gathering failed: {e}")
            general_context['system'] = {'error': str(e)}

        temporal_context = self._get_temporal_context(context)
        general_context['temporal'] = temporal_context
        return general_context

    def _get_network_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get network context."""
        # (Original _get_network_context method code from source: 861)
        return {
            'active_connections': 0,  # Placeholder
            'bandwidth_usage': 'normal',
            'unusual_ports': [],
            'external_connections': []
        }

    def _get_system_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get system context."""
        # (Original _get_system_context method code from source: 862)
        return {
            'cpu_usage': 'normal',
            'memory_usage': 'normal',
            'disk_activity': 'normal',
            'running_processes': []  # Placeholder
        }

    def _get_temporal_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get temporal context."""
        # (Original _get_temporal_context method code from source: 863)
        now = datetime.now()
        return {
            'current_time': now.isoformat(),
            'is_business_hours': 9 <= now.hour <= 17 and now.weekday() < 5,  # Mon-Fri, 9-5
            'day_of_week': now.strftime('%A'),
            'is_holiday': False  # Placeholder, needs holiday calendar logic
        }

    def _analyze_investigation_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across all investigations."""
        # (Original _analyze_investigation_patterns method code from source: 864 to source: 869)
        patterns = {
            'risk_indicators': [],
            'correlations': [],
            'anomalies': []
        }
        for entity_type, entities_list in results.get('entities', {}).items():
            for entity_details in entities_list:
                investigation_data = entity_details.get('investigation', {})
                # Example: Check IP reputation for risk
                if entity_type == 'ip' and investigation_data.get('reputation', {}).get('is_malicious'):
                    patterns['risk_indicators'].append({
                        'type': 'malicious_ip',
                        'value': entity_details.get('value'),
                        'details': investigation_data['reputation']
                    })
                # Example: Check URL reputation
                if entity_type == 'url' and investigation_data.get('reputation', {}).get('is_malicious'):
                    patterns['risk_indicators'].append({
                        'type': 'malicious_url',
                        'value': entity_details.get('value'),
                        'details': investigation_data['reputation']
                    })
        # Simplified correlation: if a malicious IP and URL are found
        if any(ind['type'] == 'malicious_ip' for ind in patterns['risk_indicators']) and \
                any(ind['type'] == 'malicious_url' for ind in patterns['risk_indicators']):
            patterns['correlations'].append({
                'type': 'malicious_ip_and_url_present',
                'severity': 'high'
            })
        return patterns

    def _generate_investigation_features(self,
                                         results: Dict[str, Any],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate features based on investigation results."""
        # (Original _generate_investigation_features method code from source: 870 to source: 876)
        features = {}
        risk_score = 0.0
        risk_factors_count = 0

        for entity_type, entities_list in results.get('entities', {}).items():
            for entity_details in entities_list:
                investigation_data = entity_details.get('investigation', {})
                if investigation_data.get('reputation', {}).get('is_malicious'):
                    risk_score += 0.3
                    risk_factors_count += 1
                current_risk = investigation_data.get('reputation', {}).get('risk_score', 0)
                if isinstance(current_risk, (int, float)) and current_risk > 0.7:  # Check type
                    risk_score += 0.2
                    risk_factors_count += 1

        if risk_factors_count > 0:
            risk_score = min(1.0,
                             risk_score / risk_factors_count if risk_factors_count > 0 else 0)  # Averaged and capped

        features['overall_risk_score'] = risk_score
        features['risk_factors_count'] = risk_factors_count

        temporal_context = results.get('context', {}).get('temporal', {})
        features['is_business_hours'] = temporal_context.get('is_business_hours', True)
        features['day_of_week'] = temporal_context.get('day_of_week', 'Unknown')

        investigation_patterns = results.get('patterns', {})
        features['risk_indicators_count'] = len(investigation_patterns.get('risk_indicators', []))
        features['correlations_found'] = len(investigation_patterns.get('correlations', [])) > 0
        features['anomalies_detected'] = len(investigation_patterns.get('anomalies', [])) > 0

        return features