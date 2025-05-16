# src/neural_components/context_gathering/investigation_strategies/ip_investigator_strategy.py
import logging
import socket
from datetime import datetime
from typing import Dict, Any

# Assuming _get_from_cache, _set_cache, _get_ip_geolocation, _check_ip_reputation, _get_network_info
# are either part of a base class, utility functions, or will be passed/accessed differently.
# For this example, let's assume they are part of a utility or base class the main investigator provides.

logger = logging.getLogger(__name__)

class IPInvestigationStrategy:
    def __init__(self, cache_manager, api_helpers):
        """
        Args:
            cache_manager: An object with get_from_cache and set_cache methods.
            api_helpers: An object or dict containing helper methods for API calls
                         (e.g., _get_ip_geolocation, _check_ip_reputation, _get_network_info).
        """
        self.cache_manager = cache_manager
        self.api_helpers = api_helpers

    def investigate_ip(self, ip_address: str) -> Dict[str, Any]:
        """
        Investigate an IP address.
        Args:
            ip_address: IP address to investigate
        Returns:
            IP investigation results
        """
        cache_key = f"ip_{ip_address}"
        cached_result = self.cache_manager._get_from_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for IP: {ip_address}")
            return cached_result

        logger.debug(f"Investigating IP: {ip_address}")
        ip_info = {
            'ip': ip_address,
            'timestamp': datetime.now().isoformat()
        }

        # Geolocation lookup
        try:
            geo_info = self.api_helpers._get_ip_geolocation(ip_address)
            ip_info['geolocation'] = geo_info
        except Exception as e:
            logger.warning(f"Geolocation lookup failed for {ip_address}: {e}")
            ip_info['geolocation'] = {'error': str(e)}

        # Reputation check
        try:
            reputation = self.api_helpers._check_ip_reputation(ip_address)
            ip_info['reputation'] = reputation
        except Exception as e:
            logger.warning(f"Reputation check failed for {ip_address}: {e}")
            ip_info['reputation'] = {'error': str(e)}

        # Reverse DNS
        try:
            reverse_dns = socket.gethostbyaddr(ip_address) # socket is a standard library
            ip_info['reverse_dns'] = reverse_dns[0]
        except Exception as e:
            logger.warning(f"Reverse DNS failed for {ip_address}: {e}")
            ip_info['reverse_dns'] = {'error': str(e)} # Store error instead of None for clarity

        # Network information
        try:
            network_info = self.api_helpers._get_network_info(ip_address)
            ip_info['network'] = network_info
        except Exception as e:
            logger.warning(f"Network info lookup failed for {ip_address}: {e}")
            ip_info['network'] = {'error': str(e)}

        self.cache_manager._set_cache(cache_key, ip_info)
        return ip_info