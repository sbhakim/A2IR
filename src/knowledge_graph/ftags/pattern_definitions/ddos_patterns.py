# src/knowledge_graph/ftags/pattern_definitions/ddos_patterns.py

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.knowledge_graph.ftags.ftag_construct import FuzzyTemporalAttackGraph

logger = logging.getLogger(__name__)

def add_ddos_patterns_to_ftag(ftag: 'FuzzyTemporalAttackGraph'):
    """
    Adds Distributed Denial of Service (DDoS) attack patterns, steps,
    and indicators to the Fuzzy Temporal Attack Graph (FTAG).

    Args:
        ftag: The FuzzyTemporalAttackGraph instance to populate.
    """
    logger.info("Adding DDoS attack patterns to FTAG...")

    # --- Main DDoS Attack Pattern Node ---
    ddos_pattern_id = "ddos_attack_pattern"
    ftag.add_node(ddos_pattern_id, "attack_pattern", {
        "name": "Distributed Denial of Service (DDoS)",
        "description": "An attempt to make an online service unavailable by overwhelming it with traffic from multiple sources.",
        "mitre_technique": "T1498", # Network Denial of Service
        "severity": "high",
        "common_targets": ["web_server", "dns_server", "network_infrastructure"]
    })

    # --- DDoS Attack Steps/Phases ---
    # Phase 1: Reconnaissance (Optional but common for targeted DDoS)
    step_recon_id = "ddos_reconnaissance"
    ftag.add_node(step_recon_id, "attack_step", {
        "name": "DDoS Target Reconnaissance",
        "description": "Attacker gathers information about the target system or network to identify vulnerabilities or optimal attack vectors for a DDoS.",
        "indicators": ["port_scan_activity", "service_enumeration_logs", "vulnerability_scanning_traffic"],
        "order": 1
    })
    ftag.add_edge(ddos_pattern_id, step_recon_id, confidence=0.6, temporal_constraint=3600.0 * 24, relation_type="has_potential_step") # 24 hours prior

    # Phase 2: Resource Acquisition / Botnet Mobilization
    step_mobilization_id = "ddos_botnet_mobilization"
    ftag.add_node(step_mobilization_id, "attack_step", {
        "name": "DDoS Resource Mobilization",
        "description": "Attacker acquires or activates compromised systems (botnet) to launch the attack.",
        "indicators": ["spike_in_bot_c2_traffic", "compromised_iot_device_activity", "malware_propagation_for_botnet"],
        "order": 2
    })
    ftag.add_edge(step_recon_id, step_mobilization_id, confidence=0.7, temporal_constraint=3600.0 * 6, relation_type="precedes") # 6 hours

    # Phase 3: Attack Launch
    step_launch_id = "ddos_attack_launch"
    ftag.add_node(step_launch_id, "attack_step", {
        "name": "DDoS Attack Launch",
        "description": "The coordinated flood of traffic or requests is initiated against the target.",
        "indicators": ["sudden_traffic_volume_increase", "high_connection_rate_from_multiple_sources", "protocol_anomaly_detected"],
        "order": 3
    })
    ftag.add_edge(step_mobilization_id, step_launch_id, confidence=0.9, temporal_constraint=3600.0, relation_type="precedes") # 1 hour

    # Phase 4: Impact / Service Disruption
    step_impact_id = "ddos_service_disruption"
    ftag.add_node(step_impact_id, "attack_step", {
        "name": "Service Disruption Achieved",
        "description": "The target service becomes slow, unresponsive, or completely unavailable due to the attack.",
        "indicators": ["high_server_latency", "service_unavailability_alerts", "resource_exhaustion_on_target"],
        "order": 4
    })
    ftag.add_edge(step_launch_id, step_impact_id, confidence=0.95, temporal_constraint=600.0, relation_type="results_in") # 10 minutes

    # --- Specific DDoS Techniques (Sub-patterns or related patterns) ---

    # Volumetric Attacks
    volumetric_id = "volumetric_ddos"
    ftag.add_node(volumetric_id, "attack_technique", {
        "name": "Volumetric DDoS Attack",
        "description": "Consumes the bandwidth of the target site or service.",
        "examples": ["UDP Flood", "ICMP Flood", "DNS Amplification"],
        "mitre_subtechnique": None # General T1498
    })
    ftag.add_edge(volumetric_id, ddos_pattern_id, confidence=0.8, temporal_constraint=0, relation_type="is_type_of")
    ftag.add_edge(volumetric_id, step_launch_id, confidence=0.8, temporal_constraint=0, relation_type="manifests_as_step")

    # Protocol Attacks
    protocol_id = "protocol_ddos"
    ftag.add_node(protocol_id, "attack_technique", {
        "name": "Protocol DDoS Attack",
        "description": "Consumes server resources or resources of intermediate communication equipment, like firewalls and load balancers.",
        "examples": ["SYN Flood", "Ping of Death", "Fragmented Packet Attacks"],
        "mitre_subtechnique": None # General T1498
    })
    ftag.add_edge(protocol_id, ddos_pattern_id, confidence=0.8, temporal_constraint=0, relation_type="is_type_of")
    ftag.add_edge(protocol_id, step_launch_id, confidence=0.8, temporal_constraint=0, relation_type="manifests_as_step")


    # Application Layer Attacks
    app_layer_id = "application_layer_ddos"
    ftag.add_node(app_layer_id, "attack_technique", {
        "name": "Application Layer DDoS Attack",
        "description": "Targets specific application vulnerabilities to crash or overwhelm the application.",
        "examples": ["HTTP Flood", "Slowloris", "SQL Injection based DoS"],
        "mitre_subtechnique": "T1499.002" # Endpoint Denial of Service: Service Exhaustion Flood (could be more specific)
    })
    ftag.add_edge(app_layer_id, ddos_pattern_id, confidence=0.8, temporal_constraint=0, relation_type="is_type_of")
    ftag.add_edge(app_layer_id, step_launch_id, confidence=0.8, temporal_constraint=0, relation_type="manifests_as_step")


    # --- Common Indicators for DDoS ---
    indicator_traffic_spike_id = "indicator_ddos_traffic_spike"
    ftag.add_node(indicator_traffic_spike_id, "indicator", {
        "name": "Anomalous Traffic Spike",
        "description": "Unusually high volume of incoming network traffic.",
        "observed_in": ["firewall_logs", "network_monitoring_tools", "server_logs"],
        "severity_contribution": "high"
    })
    ftag.add_edge(indicator_traffic_spike_id, step_launch_id, confidence=0.9, temporal_constraint=60.0, relation_type="is_indicator_of")

    indicator_syn_flood_id = "indicator_syn_flood"
    ftag.add_node(indicator_syn_flood_id, "indicator", {
        "name": "SYN Flood Indicators",
        "description": "Large number of SYN packets without corresponding ACKs, half-open connections.",
        "observed_in": ["netflow_data", "packet_captures"],
        "severity_contribution": "high"
    })
    ftag.add_edge(indicator_syn_flood_id, protocol_id, confidence=0.9, temporal_constraint=60.0, relation_type="is_indicator_of")


    indicator_resource_exhaust_id = "indicator_resource_exhaustion"
    ftag.add_node(indicator_resource_exhaust_id, "indicator", {
        "name": "Server Resource Exhaustion",
        "description": "High CPU, memory, or network interface utilization on target servers.",
        "observed_in": ["server_performance_metrics", "monitoring_alerts"],
        "severity_contribution": "critical"
    })
    ftag.add_edge(indicator_resource_exhaust_id, step_impact_id, confidence=0.95, temporal_constraint=120.0, relation_type="is_indicator_of")

    logger.debug("DDoS attack patterns added to FTAG.")