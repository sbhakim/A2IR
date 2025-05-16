# src/knowledge_graph/ftags/pattern_definitions/phishing_patterns.py
import logging
from ..ftag_construct import FuzzyTemporalAttackGraph # Relative import

logger = logging.getLogger(__name__)

def add_phishing_patterns_to_ftag(ftag: FuzzyTemporalAttackGraph):
    """Add phishing attack patterns to the FTAG."""
    logger.info("Adding phishing patterns to FTAG...")

    # Main phishing pattern node
    ftag.add_node("phishing_campaign", "attack_pattern", {
        "name": "Phishing Campaign",
        "description": "Email-based attack to steal credentials or deliver malware",
        "mitre_technique": "T1566",
        "severity": "high"
    })

    # Phishing steps
    steps = [
        {
            "id": "phishing_email_sent", "name": "Phishing Email Sent",
            "description": "Malicious email sent to target",
            "indicators": ["suspicious_sender", "spoofed_domain", "urgent_language"], "confidence": 0.9
        },
        {
            "id": "email_opened", "name": "Email Opened",
            "description": "Target opens the phishing email",
            "indicators": ["email_opened_indicator", "user_interaction"], "confidence": 0.8
        },
        {
            "id": "link_clicked", "name": "Malicious Link Clicked",
            "description": "Target clicks on malicious link",
            "indicators": ["url_click", "external_domain", "suspicious_url_pattern"], "confidence": 0.7
        },
        {
            "id": "credentials_harvested", "name": "Credentials Harvested",
            "description": "Target enters credentials on fake page",
            "indicators": ["credential_submission", "form_data_exfiltration"], "confidence": 0.6
        }
    ]

    previous_step_id = "phishing_campaign"
    for i, step_data in enumerate(steps):
        step_id = step_data["id"]
        ftag.add_node(step_id, "attack_step", {
            "name": step_data["name"], "description": step_data["description"],
            "indicators": step_data["indicators"], "order": i + 1
        })
        ftag.add_edge(
            previous_step_id, step_id,
            confidence=step_data["confidence"],
            temporal_constraint=300.0 if i == 0 else 1800.0,
            relation_type="precedes"
        )
        previous_step_id = step_id

    # Add phishing subtypes
    subtypes = [
        ("spear_phishing", "Spear Phishing", "Targeted phishing attack", "T1566.001"),
        ("whaling", "Whaling", "Executive-targeted phishing", "T1566.002"),
        ("attachment_phishing", "Attachment Phishing", "Malicious attachment delivery", "T1566.001")
    ]
    for subtype_id, name, desc, mitre in subtypes:
        ftag.add_node(subtype_id, "attack_pattern", {
            "name": name, "description": desc,
            "mitre_technique": mitre, "severity": "high"
        })
        ftag.add_edge(
            subtype_id, "phishing_campaign",
            confidence=0.95, temporal_constraint=0.0,
            relation_type="is_subtype_of"
        )
    logger.debug("Phishing patterns added.")