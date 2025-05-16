# src/knowledge_graph/ftags/ftag_initializer.py

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Ensure correct relative import for FuzzyTemporalAttackGraph
# If ftag_construct is in the same directory (ftags):
from .ftag_construct import FuzzyTemporalAttackGraph

# If pattern_definitions is a sub-package of ftags:
from .pattern_definitions.phishing_patterns import add_phishing_patterns_to_ftag
from .pattern_definitions.malware_patterns import add_malware_patterns_to_ftag
from .pattern_definitions.ddos_patterns import add_ddos_patterns_to_ftag
from .pattern_definitions.insider_threat_patterns import add_insider_threat_patterns_to_ftag
from .pattern_definitions.common_indicator_patterns import add_common_indicators_to_ftag
from .pattern_definitions.relationship_patterns import add_cross_pattern_relationships_to_ftag
from .pattern_definitions.temporal_definitions import add_temporal_constraints_to_ftag

# Add imports for other pattern files you create (portscan, web_attack, etc.)
# from .pattern_definitions.portscan_patterns import add_portscan_patterns_to_ftag
# from .pattern_definitions.web_attack_patterns import add_web_attack_patterns_to_ftag
# from .pattern_definitions.infiltration_patterns import add_infiltration_patterns_to_ftag


logger = logging.getLogger(__name__)


class FTAGInitializer:
    """
    Initializes the Fuzzy Temporal Attack Graph with threat intelligence data.
    This component populates the knowledge graph with:
    - Attack patterns from custom definitions
    - Techniques, tactics, etc., from MITRE ATT&CK (via STIX)
    - Common security indicators
    - Relationships between threats and techniques
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the FTAG initializer.
        Args:
            config: Configuration dictionary, expected to be the 'knowledge_graph'
                    section from the main config.
        """
        self.config = config or {}
        # Default path for MITRE data if not specified in config
        # Ensure this path points to your downloaded STIX bundle (e.g., ics-attack.json)
        self.mitre_data_path = self.config.get('mitre_data_path', 'data/knowledge_base/mitre_attack.json')
        # self.indicators_path = self.config.get('indicators_path', 'data/knowledge_base/indicators.json') # If you have separate indicators file
        logger.info(f"FTAGInitializer initialized. MITRE data path configured to: {self.mitre_data_path}")

    def initialize_ftag(self, ftag: FuzzyTemporalAttackGraph) -> bool:
        """
        Initialize an FTAG with base threat intelligence.
        Args:
            ftag: The FTAG instance to initialize
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting FTAG initialization...")

            # Add custom-defined core attack patterns
            add_phishing_patterns_to_ftag(ftag)
            add_malware_patterns_to_ftag(ftag)  # Make sure this file and function exist
            add_ddos_patterns_to_ftag(ftag)  # Make sure this file and function exist
            add_insider_threat_patterns_to_ftag(ftag)  # Make sure this file and function exist
            # Call other custom pattern functions here:
            # add_portscan_patterns_to_ftag(ftag)
            # add_web_attack_patterns_to_ftag(ftag)
            # add_infiltration_patterns_to_ftag(ftag)

            # Add common indicators
            add_common_indicators_to_ftag(ftag)  # Make sure this file and function exist

            # Add relationships between patterns
            add_cross_pattern_relationships_to_ftag(ftag)  # Make sure this file and function exist

            # Add temporal constraints
            add_temporal_constraints_to_ftag(ftag)  # Make sure this file and function exist

            # Load MITRE data from STIX bundle
            self.load_from_mitre_stix_bundle(ftag)

            logger.info("FTAG initialization complete.")
            return True

        except Exception as e:
            logger.error(f"Error initializing FTAG: {e}", exc_info=True)
            return False

    def load_from_mitre_stix_bundle(self, ftag: FuzzyTemporalAttackGraph) -> bool:
        """
        Load MITRE ATT&CK data from a STIX 2.x JSON bundle into the FTAG.
        This method is designed to parse standard MITRE ATT&CK STIX bundles.

        Args:
            ftag: The FTAG instance to populate.
        Returns:
            bool: True if successful, False otherwise.
        """
        if not os.path.exists(self.mitre_data_path):
            logger.warning(f"MITRE STIX data file not found: {self.mitre_data_path}")
            return False
        try:
            with open(self.mitre_data_path, 'r', encoding='utf-8') as f:
                stix_bundle = json.load(f)

            if not isinstance(stix_bundle, dict) or stix_bundle.get("type") != "bundle" or "objects" not in stix_bundle:
                logger.error(f"File {self.mitre_data_path} does not appear to be a valid STIX bundle.")
                return False

            mitre_objects = stix_bundle.get('objects', [])
            tactics_map = {}  # To store tactic names by their STIX ID

            # First pass: Collect tactics to resolve names later
            for stix_object in mitre_objects:
                if stix_object.get('type') == 'x-mitre-tactic':
                    tactic_id = stix_object.get('id')
                    tactic_name = stix_object.get('name')
                    # Extract short name from x_mitre_shortname if available, else parse from ID
                    tactic_shortname = stix_object.get('x_mitre_shortname', tactic_name.lower().replace(' ', '-'))
                    if tactic_id and tactic_name:
                        tactics_map[tactic_id] = {
                            "name": tactic_name,
                            "shortname": tactic_shortname
                        }
                        # Add tactic node to FTAG
                        ftag.add_node(tactic_shortname, "mitre_tactic", {  # Use shortname as ID for simplicity
                            "name": tactic_name,
                            "description": stix_object.get('description', ''),
                            "stix_id": tactic_id
                        })

            # Second pass: Process attack patterns (techniques) and other objects
            techniques_added = 0
            for stix_object in mitre_objects:
                obj_type = stix_object.get('type')

                if obj_type == 'attack-pattern':  # This is a MITRE Technique
                    technique_stix_id = stix_object.get('id')
                    technique_name = stix_object.get('name')
                    technique_attck_id = None

                    # Find the ATT&CK ID (e.g., T1566) from external_references
                    for ref in stix_object.get('external_references', []):
                        if ref.get('source_name') in ['mitre-attack', 'mitre-mobile-attack', 'mitre-ics-attack']:
                            technique_attck_id = ref.get('external_id')
                            break

                    if not technique_attck_id:  # Fallback if specific ATT&CK ID not found
                        technique_attck_id = technique_name.replace(' ', '_').upper()[:15]  # Create a placeholder ID

                    if technique_attck_id and technique_name:
                        tactic_names_for_technique = []
                        if 'kill_chain_phases' in stix_object:
                            for phase in stix_object['kill_chain_phases']:
                                # Make sure kill_chain_name corresponds to a known MITRE ATT&CK kill chain
                                if phase.get('kill_chain_name') in ['mitre-attack', 'mitre-enterprise-attack',
                                                                    'mitre-mobile-attack', 'mitre-ics-attack']:
                                    tactic_shortname_ref = phase.get(
                                        'phase_name')  # This is usually the shortname like 'initial-access'
                                    # Find the full tactic name if needed, or just store shortname
                                    tactic_names_for_technique.append(tactic_shortname_ref)

                        node_attrs = {
                            "name": technique_name,
                            "description": stix_object.get('description', ''),
                            "tactics": tactic_names_for_technique,  # List of tactic shortnames
                            "platforms": stix_object.get('x_mitre_platforms', []),
                            "stix_id": technique_stix_id,
                            "is_subtechnique": stix_object.get('x_mitre_is_subtechnique', False)
                        }
                        if stix_object.get('x_mitre_deprecated'):
                            node_attrs['deprecated'] = True

                        ftag.add_node(technique_attck_id, "mitre_technique", node_attrs)
                        techniques_added += 1

                        # Link technique to its tactics
                        for tactic_shortname in tactic_names_for_technique:
                            if ftag.graph.has_node(tactic_shortname):  # Check if tactic node was created
                                ftag.add_edge(tactic_shortname, technique_attck_id,
                                              confidence=1.0, temporal_constraint=0, relation_type="includes_technique")

                        # Link sub-techniques to parent techniques
                        if node_attrs["is_subtechnique"]:
                            # STIX uses a 'subtechnique-of' relationship object. We need to find it.
                            # This part is more complex as it requires finding relationship objects.
                            # For a simpler approach, if parent ID is in external_references of subtechnique:
                            for ref in stix_object.get('external_references', []):
                                if ref.get('source_name') in ['mitre-attack', 'mitre-mobile-attack',
                                                              'mitre-ics-attack']:
                                    # Example: T1548.002 -> external_id might contain T1548.002, description might mention T1548
                                    # A more robust way is to look for relationship objects of type 'subtechnique-of'
                                    # Or if ATT&CK ID is like "Txxxx.xxx", the "Txxxx" is the parent.
                                    if '.' in technique_attck_id:
                                        parent_attck_id = technique_attck_id.split('.')[0]
                                        if ftag.graph.has_node(parent_attck_id):
                                            ftag.add_edge(parent_attck_id, technique_attck_id,
                                                          confidence=1.0, temporal_constraint=0,
                                                          relation_type="has_subtechnique")
                                        else:
                                            logger.debug(
                                                f"Parent technique {parent_attck_id} for {technique_attck_id} not found yet.")

                # TODO: Optionally process other STIX object types like 'malware', 'tool', 'intrusion-set' (campaigns)
                # and link them to techniques they use via 'relationship' objects.
                # This would involve iterating again for 'relationship' objects and connecting nodes.

            logger.info(f"Loaded {techniques_added} MITRE ATT&CK techniques from STIX bundle: {self.mitre_data_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding MITRE STIX JSON from {self.mitre_data_path}: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Error loading MITRE STIX data: {e}", exc_info=True)
            return False


# Convenience function to be called by the orchestrator
def initialize_knowledge_graph(config: Dict[str, Any]) -> Optional[FuzzyTemporalAttackGraph]:
    """
    Convenience function to create and initialize an FTAG.
    The 'config' passed here should be the main application configuration.
    Args:
        config: Main application configuration.
    Returns:
        Initialized FTAG instance or None if failed.
    """
    kg_specific_config = config.get('knowledge_graph', {})  # Extract KG specific config

    # Pass the KG specific config to FuzzyTemporalAttackGraph if its __init__ expects it
    # Current ftag_construct.py's __init__ only takes 'name'. If it's updated to take config:
    # ftag = FuzzyTemporalAttackGraph(
    #     name=kg_specific_config.get("name", "a2ir_knowledge_base"),
    #     config=kg_specific_config
    # )
    # Based on current ftag_construct.py:
    ftag = FuzzyTemporalAttackGraph(name=kg_specific_config.get("name", "a2ir_knowledge_base"))

    # Pass the KG specific config to FTAGInitializer
    initializer = FTAGInitializer(kg_specific_config)
    if initializer.initialize_ftag(ftag):
        logger.info("Knowledge graph initialized successfully via convenience function.")
        return ftag
    else:
        logger.error("Failed to initialize knowledge graph via convenience function.")
        return None