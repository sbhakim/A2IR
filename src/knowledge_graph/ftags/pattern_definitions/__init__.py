# src/knowledge_graph/ftags/pattern_definitions/__init__.py

from .phishing_patterns import add_phishing_patterns_to_ftag
from .malware_patterns import add_malware_patterns_to_ftag
from .ddos_patterns import add_ddos_patterns_to_ftag
from .insider_threat_patterns import add_insider_threat_patterns_to_ftag
from .common_indicator_patterns import add_common_indicators_to_ftag
from .relationship_patterns import add_cross_pattern_relationships_to_ftag
from .temporal_definitions import add_temporal_constraints_to_ftag

__all__ = [
    "add_phishing_patterns_to_ftag",
    "add_malware_patterns_to_ftag",
    "add_ddos_patterns_to_ftag",
    "add_insider_threat_patterns_to_ftag",
    "add_common_indicators_to_ftag",
    "add_cross_pattern_relationships_to_ftag",
    "add_temporal_constraints_to_ftag",
]