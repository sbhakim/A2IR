# src/neural_components/context_gathering/investigation_strategies/url_investigator_strategy.py
import logging
from typing import Dict, Any

import requests # Make sure this is installed
from urllib.parse import urlparse
from datetime import datetime
# Assuming _get_from_cache, _set_cache, _check_url_reputation, _check_url_alive
# and _investigate_domain are part of a base class, utility functions, or the main investigator.

logger = logging.getLogger(__name__)

class URLInvestigationStrategy:
    def __init__(self, cache_manager, api_helpers, domain_investigator):
        self.cache_manager = cache_manager
        self.api_helpers = api_helpers
        self.domain_investigator = domain_investigator # To investigate the domain part of the URL

    def investigate_url(self, url: str) -> Dict[str, Any]:
        """
        Investigate a URL.
        Args:
            url: URL to investigate
        Returns:
            URL investigation results
        """
        cache_key = f"url_{url}"
        cached_result = self.cache_manager._get_from_cache(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for URL: {url}")
            return cached_result

        logger.debug(f"Investigating URL: {url}")
        url_info = {
            'url': url,
            'timestamp': datetime.now().isoformat()
        }

        # Parse URL components
        try:
            parsed = urlparse(url)
            url_info['components'] = {
                'scheme': parsed.scheme,
                'domain': parsed.netloc,
                'path': parsed.path,
                'params': parsed.params,
                'query': parsed.query
            }

            # Investigate the domain part
            if parsed.netloc:
                # Assuming domain_investigator has an investigate_domain method
                domain_data = self.domain_investigator.investigate_domain(parsed.netloc)
                url_info['domain_info'] = domain_data

        except Exception as e:
            logger.warning(f"URL parsing failed for {url}: {e}")
            url_info['components'] = {'error': str(e)}

        # Check URL reputation
        try:
            reputation = self.api_helpers._check_url_reputation(url)
            url_info['reputation'] = reputation
        except Exception as e:
            logger.warning(f"URL reputation check failed for {url}: {e}")
            url_info['reputation'] = {'error': str(e)}

        # Check if URL is alive
        try:
            is_alive = self.api_helpers._check_url_alive(url)
            url_info['is_alive'] = is_alive
        except Exception as e:
            logger.warning(f"URL aliveness check failed for {url}: {e}")
            url_info['is_alive'] = {'error': str(e)}

        self.cache_manager._set_cache(cache_key, url_info)
        return url_info