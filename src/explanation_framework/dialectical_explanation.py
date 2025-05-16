# src/explanation_framework/dialectical_explanation.py

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DialecticalExplanation:
    """
    Implements the dialectical explanation framework for A²IR.

    This models explanations as an iterative conversation between neural and symbolic 
    components, following the formalization:
    Explanation = {(ni, si)}_{i=1}^k

    Where ni and si are neural and symbolic contributions at step i.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dialectical explanation generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 5)
        self.verbosity_level = self.config.get('verbosity_level', 'medium')

        # State tracking
        self.current_state = None
        self.explanation_history = []

        logger.info("Dialectical explanation framework initialized")

    def generate_dialectical_explanation(self,
                                         incident_data: Dict[str, Any],
                                         integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a dialectical explanation for an incident.

        Args:
            incident_data: Original incident/alert data
            integration_result: Result from neurosymbolic integration

        Returns:
            Dialectical explanation with neural-symbolic dialogue
        """
        self.current_state = self._initialize_state(incident_data, integration_result)
        dialectical_steps = []

        # Extract components from integration result
        classification = integration_result.get('classification', {})
        threats = integration_result.get('identified_threats', [])
        actions = integration_result.get('recommended_actions', [])
        reasoning_path = integration_result.get('explanation', {}).get('reasoning_path', [])
        integration_trace = integration_result.get('explanation', {}).get('integration_trace', [])

        # Step 1: Neural feature extraction (n1)
        n1 = self._neural_step_1_feature_extraction(incident_data, classification)
        dialectical_steps.append(('neural', n1))
        self._update_state('features', n1['extracted_features'])

        # Step 2: Symbolic pattern matching (s1)
        s1 = self._symbolic_step_1_pattern_matching(threats, reasoning_path)
        dialectical_steps.append(('symbolic', s1))
        self._update_state('patterns', s1['matched_patterns'])

        # Step 3: Neural narrative generation (n2)
        n2 = self._neural_step_2_narrative_generation(threats, classification)
        dialectical_steps.append(('neural', n2))
        self._update_state('narrative', n2['narrative'])

        # Step 4: Symbolic justification (s2)
        s2 = self._symbolic_step_2_justification(actions, reasoning_path)
        dialectical_steps.append(('symbolic', s2))
        self._update_state('justifications', s2['justifications'])

        # Step 5: Neural synthesis (n3)
        n3 = self._neural_step_3_synthesis(dialectical_steps)
        dialectical_steps.append(('neural', n3))
        self._update_state('synthesis', n3['synthesis'])

        # Generate final explanation
        explanation = self._generate_final_explanation(
            dialectical_steps,
            integration_result,
            incident_data
        )

        # Store in history
        self.explanation_history.append({
            'incident_id': integration_result.get('incident_id'),
            'timestamp': datetime.now().isoformat(),
            'explanation': explanation
        })

        return explanation

    def _initialize_state(self, incident_data: Dict[str, Any],
                          integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the state for dialectical process."""
        return {
            'features': {},
            'patterns': [],
            'narrative': "",
            'justifications': [],
            'synthesis': "",
            'incident_type': integration_result.get('classification', {}).get('type'),
            'confidence': integration_result.get('classification', {}).get('confidence', 0)
        }

    def _update_state(self, key: str, value: Any):
        """Update the current state."""
        if self.current_state:
            self.current_state[key] = value

    def _neural_step_1_feature_extraction(self,
                                          incident_data: Dict[str, Any],
                                          classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Neural Step 1: Extract salient features from the alert.
        """
        features_text = "The alert indicates "
        extracted_features = []

        # Extract key features based on alert type
        alert_type = classification.get('type', 'unknown')
        alert_data = incident_data.get('data', {})

        if alert_type == 'phishing':
            if 'email' in alert_data:
                email_data = alert_data['email']
                if 'sender' in email_data:
                    features_text += f"an email from '{email_data['sender']}' "
                    extracted_features.append(f"sender: {email_data['sender']}")

                if 'subject' in email_data:
                    features_text += f"with subject '{email_data['subject']}' "
                    extracted_features.append(f"subject: {email_data['subject']}")

                if 'urls' in email_data and email_data['urls']:
                    features_text += f"containing {len(email_data['urls'])} URL(s) "
                    extracted_features.append(f"urls: {email_data['urls']}")

        elif alert_type == 'malware':
            if 'malware' in alert_data:
                malware_data = alert_data['malware']
                if 'processes' in malware_data:
                    features_text += f"suspicious process activity: {malware_data['processes']} "
                    extracted_features.append(f"processes: {malware_data['processes']}")

                if 'hashes' in malware_data:
                    features_text += f"with file hashes: {', '.join(malware_data['hashes'][:2])}... "
                    extracted_features.append(f"hashes: {malware_data['hashes']}")

        else:
            # Generic feature extraction
            if 'source_ip' in alert_data:
                features_text += f"activity from IP {alert_data['source_ip']} "
                extracted_features.append(f"source_ip: {alert_data['source_ip']}")

            if 'dest_ip' in alert_data:
                features_text += f"targeting IP {alert_data['dest_ip']} "
                extracted_features.append(f"dest_ip: {alert_data['dest_ip']}")

        # Add confidence
        confidence = classification.get('confidence', 0)
        features_text += f"(classification confidence: {confidence:.1%})"

        return {
            'step': 'neural_feature_extraction',
            'description': features_text,
            'extracted_features': extracted_features,
            'timestamp': datetime.now().isoformat()
        }

    def _symbolic_step_1_pattern_matching(self,
                                          threats: List[Dict[str, Any]],
                                          reasoning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Symbolic Step 1: Match features to known attack patterns.
        """
        pattern_text = "These features match "
        matched_patterns = []

        if threats:
            # Describe identified threats
            threat = threats[0]  # Focus on primary threat
            pattern_text += f"pattern '{threat.get('type', 'unknown')}' "
            pattern_text += f"(ID: {threat.get('id', 'unknown')}) "
            pattern_text += f"with {threat.get('confidence', 0):.1%} confidence. "

            matched_patterns.append({
                'pattern_id': threat.get('id'),
                'pattern_type': threat.get('type'),
                'confidence': threat.get('confidence', 0)
            })

            # Add reasoning path information
            if reasoning_path:
                pattern_text += "The reasoning path includes: "
                for i, step in enumerate(reasoning_path[:3]):  # Limit to first 3 steps
                    if i > 0:
                        pattern_text += ", "
                    pattern_text += f"{step.get('description', step.get('rule_id', 'unknown'))}"
                pattern_text += "."
        else:
            pattern_text += "no known attack patterns with high confidence."

        return {
            'step': 'symbolic_pattern_matching',
            'description': pattern_text,
            'matched_patterns': matched_patterns,
            'reasoning_path': reasoning_path[:3] if reasoning_path else [],
            'timestamp': datetime.now().isoformat()
        }

    def _neural_step_2_narrative_generation(self,
                                            threats: List[Dict[str, Any]],
                                            classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Neural Step 2: Generate natural language description of the threat.
        """
        narrative = ""
        alert_type = classification.get('type', 'unknown')

        # Generate narrative based on alert type
        if alert_type == 'phishing':
            narrative = ("The system has identified behavior consistent with a phishing attack. "
                         "This type of attack typically aims to steal credentials or deliver malware "
                         "by tricking users into clicking malicious links or opening harmful attachments.")

        elif alert_type == 'malware':
            narrative = ("The system has detected activity indicative of malware infection. "
                         "The identified processes and behaviors suggest malicious software may be "
                         "attempting to establish persistence, communicate with command servers, "
                         "or perform unauthorized actions on the system.")

        elif alert_type == 'ddos':
            narrative = (
                "The system has identified patterns consistent with a Distributed Denial of Service (DDoS) attack. "
                "This involves overwhelming the target with traffic to disrupt normal operations.")

        elif alert_type == 'insider_threat':
            narrative = ("The system has detected anomalous user behavior that may indicate an insider threat. "
                         "This could involve unauthorized data access, exfiltration attempts, or privilege abuse.")

        else:
            narrative = (f"The system has classified this as a '{alert_type}' incident. "
                         "While the specific attack pattern is less common, the identified indicators "
                         "suggest potentially malicious activity requiring investigation.")

        # Add threat-specific details
        if threats:
            threat = threats[0]
            narrative += f" The specific threat identified is '{threat.get('type', 'unknown')}' "
            narrative += f"with {threat.get('confidence', 0):.1%} confidence."

        return {
            'step': 'neural_narrative_generation',
            'description': narrative,
            'narrative': narrative,
            'timestamp': datetime.now().isoformat()
        }

    def _symbolic_step_2_justification(self,
                                       actions: List[Dict[str, Any]],
                                       reasoning_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Symbolic Step 2: Provide logical justification for recommendations.
        """
        justification_text = "Recommended actions: "
        justifications = []

        if actions:
            for i, action in enumerate(actions):
                if i > 0:
                    justification_text += "; "

                action_type = action.get('action', 'unknown')
                target = action.get('target', 'system')
                priority = action.get('priority', 'medium')

                justification_text += f"{action_type} {target} (priority: {priority})"

                # Add justification based on action type
                if action_type == 'isolate':
                    justification_text += " - prevents lateral movement and contains potential infection"
                elif action_type == 'block_sender':
                    justification_text += " - prevents further phishing attempts from the same source"
                elif action_type == 'quarantine_email':
                    justification_text += " - removes the threat from user's inbox while preserving evidence"

                justifications.append({
                    'action': action_type,
                    'target': target,
                    'priority': priority,
                    'reasoning': self._get_action_reasoning(action_type, reasoning_path)
                })
        else:
            justification_text += "Monitor the situation and gather more information."
            justifications.append({
                'action': 'monitor',
                'target': 'system',
                'priority': 'low',
                'reasoning': 'Insufficient evidence for immediate action'
            })

        # Add rule-based justification
        if reasoning_path:
            justification_text += " These recommendations are based on: "
            rules_used = [step.get('rule_id', step.get('description', ''))
                          for step in reasoning_path if 'rule_id' in step or 'description' in step]
            justification_text += ", ".join(rules_used[:3])

        return {
            'step': 'symbolic_justification',
            'description': justification_text,
            'justifications': justifications,
            'rules_applied': rules_used[:3] if reasoning_path else [],
            'timestamp': datetime.now().isoformat()
        }

    def _get_action_reasoning(self, action_type: str, reasoning_path: List[Dict[str, Any]]) -> str:
        """Get specific reasoning for an action type."""
        default_reasoning = {
            'isolate': "Isolation prevents spread of infection and limits damage",
            'block_sender': "Blocking prevents future attacks from the same source",
            'quarantine_email': "Quarantine removes immediate threat while preserving evidence",
            'monitor': "Monitoring allows gathering more information before taking action"
        }

        # Look for specific reasoning in the path
        for step in reasoning_path:
            if action_type in str(step):
                return step.get('description', default_reasoning.get(action_type, "Security best practice"))

        return default_reasoning.get(action_type, "Based on security best practices")

    def _neural_step_3_synthesis(self, dialectical_steps: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Neural Step 3: Synthesize a coherent narrative connecting all elements.
        """
        # Extract information from previous steps
        features = None
        patterns = None
        narrative = None
        actions = None

        for step_type, step_data in dialectical_steps:
            if step_data['step'] == 'neural_feature_extraction':
                features = step_data['extracted_features']
            elif step_data['step'] == 'symbolic_pattern_matching':
                patterns = step_data['matched_patterns']
            elif step_data['step'] == 'neural_narrative_generation':
                narrative = step_data['narrative']
            elif step_data['step'] == 'symbolic_justification':
                actions = step_data['justifications']

        # Create synthesis
        synthesis = "Based on the analysis, "

        # Summarize the threat
        if self.current_state and self.current_state.get('incident_type'):
            incident_type = self.current_state['incident_type']
            confidence = self.current_state.get('confidence', 0)
            synthesis += f"this is a {incident_type} incident with {confidence:.1%} confidence. "

        # Summarize key features
        if features and len(features) > 0:
            synthesis += f"Key indicators include {features[0]}. "

        # Connect to patterns
        if patterns and len(patterns) > 0:
            pattern = patterns[0]
            synthesis += (f"This matches the '{pattern['pattern_type']}' attack pattern "
                          f"with {pattern['confidence']:.1%} confidence. ")

        # State the risk
        if narrative:
            # Extract key risk from narrative
            risk_phrases = narrative.split('. ')
            if len(risk_phrases) > 1:
                synthesis += risk_phrases[1] + " "

        # Recommend actions
        if actions and len(actions) > 0:
            primary_action = actions[0]
            synthesis += (f"The recommended response is to {primary_action['action']} "
                          f"the {primary_action['target']} (priority: {primary_action['priority']}). ")
            synthesis += f"This action {primary_action['reasoning'].lower()}."

        return {
            'step': 'neural_synthesis',
            'description': synthesis,
            'synthesis': synthesis,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_final_explanation(self,
                                    dialectical_steps: List[Tuple[str, Dict[str, Any]]],
                                    integration_result: Dict[str, Any],
                                    incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final dialectical explanation."""
        # Extract key components
        synthesis = ""
        for step_type, step_data in dialectical_steps:
            if step_data['step'] == 'neural_synthesis':
                synthesis = step_data['synthesis']
                break

        # Structure the explanation
        explanation = {
            'incident_id': integration_result.get('incident_id'),
            'timestamp': datetime.now().isoformat(),
            'summary': synthesis,
            'dialectical_process': {
                'steps': []
            },
            'classification': integration_result.get('classification', {}),
            'threats': integration_result.get('identified_threats', []),
            'recommended_actions': integration_result.get('recommended_actions', []),
            'confidence_score': integration_result.get('classification', {}).get('confidence', 0)
        }

        # Add dialectical steps
        for i, (step_type, step_data) in enumerate(dialectical_steps):
            explanation['dialectical_process']['steps'].append({
                'step_number': i + 1,
                'type': step_type,
                'name': step_data['step'],
                'description': step_data['description'],
                'timestamp': step_data['timestamp']
            })

        # Add verbosity levels
        if self.verbosity_level == 'high':
            # Include all details
            explanation['detailed_steps'] = dialectical_steps
        elif self.verbosity_level == 'low':
            # Only include summary
            explanation = {
                'incident_id': explanation['incident_id'],
                'summary': explanation['summary'],
                'classification': explanation['classification'],
                'recommended_actions': explanation['recommended_actions']
            }

        return explanation

    def format_explanation_for_display(self, explanation: Dict[str, Any]) -> str:
        """
        Format the explanation for human-readable display.

        Args:
            explanation: The dialectical explanation

        Returns:
            Formatted string for display
        """
        formatted = []

        # Header
        formatted.append("=== A²IR Incident Analysis ===")
        formatted.append(f"Incident ID: {explanation.get('incident_id', 'Unknown')}")
        formatted.append(f"Timestamp: {explanation.get('timestamp', 'Unknown')}")
        formatted.append("")

        # Summary
        formatted.append("EXECUTIVE SUMMARY:")
        formatted.append(explanation.get('summary', 'No summary available'))
        formatted.append("")

        # Classification
        classification = explanation.get('classification', {})
        formatted.append("CLASSIFICATION:")
        formatted.append(f"Type: {classification.get('type', 'Unknown')}")
        formatted.append(f"Confidence: {classification.get('confidence', 0):.1%}")
        formatted.append("")

        # Threats
        threats = explanation.get('threats', [])
        if threats:
            formatted.append("IDENTIFIED THREATS:")
            for threat in threats:
                formatted.append(f"- {threat.get('type', 'Unknown')} (ID: {threat.get('id', 'Unknown')}, "
                                 f"Confidence: {threat.get('confidence', 0):.1%})")
            formatted.append("")

        # Recommended Actions
        actions = explanation.get('recommended_actions', [])
        if actions:
            formatted.append("RECOMMENDED ACTIONS:")
            for action in actions:
                formatted.append(f"- {action.get('action', 'Unknown')} {action.get('target', 'system')} "
                                 f"(Priority: {action.get('priority', 'Unknown')})")
            formatted.append("")

        # Dialectical Process (if included)
        if 'dialectical_process' in explanation:
            formatted.append("ANALYSIS PROCESS:")
            for step in explanation['dialectical_process']['steps']:
                formatted.append(f"{step['step_number']}. [{step['type'].upper()}] {step['name']}")
                formatted.append(f"   {step['description']}")
            formatted.append("")

        return "\n".join(formatted)

    def evaluate_explanation_quality(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the quality of a dialectical explanation.

        Args:
            explanation: The explanation to evaluate

        Returns:
            Quality scores for different dimensions
        """
        scores = {}

        # Completeness - Does it have all required components?
        required_components = ['summary', 'classification', 'threats', 'recommended_actions']
        present_components = sum(1 for comp in required_components if comp in explanation and explanation[comp])
        scores['completeness'] = present_components / len(required_components)

        # Coherence - Are the steps logically connected?
        if 'dialectical_process' in explanation:
            steps = explanation['dialectical_process']['steps']
            expected_order = ['neural_feature_extraction', 'symbolic_pattern_matching',
                              'neural_narrative_generation', 'symbolic_justification', 'neural_synthesis']
            actual_order = [step['name'] for step in steps]

            correct_order = sum(1 for i, expected in enumerate(expected_order)
                                if i < len(actual_order) and actual_order[i] == expected)
            scores['coherence'] = correct_order / len(expected_order)
        else:
            scores['coherence'] = 0.0

        # Clarity - Is the summary clear and concise?
        summary = explanation.get('summary', '')
        if summary:
            # Simple heuristics for clarity
            word_count = len(summary.split())
            sentence_count = len(summary.split('.'))
            avg_sentence_length = word_count / max(sentence_count, 1)

            # Ideal sentence length is 15-20 words
            if 15 <= avg_sentence_length <= 20:
                clarity_score = 1.0
            else:
                deviation = abs(avg_sentence_length - 17.5) / 17.5
                clarity_score = max(0.0, 1.0 - deviation)

            scores['clarity'] = clarity_score
        else:
            scores['clarity'] = 0.0

        # Actionability - Are clear actions provided?
        actions = explanation.get('recommended_actions', [])
        if actions:
            actionable_actions = sum(1 for action in actions
                                     if 'action' in action and 'target' in action and 'priority' in action)
            scores['actionability'] = actionable_actions / max(len(actions), 1)
        else:
            scores['actionability'] = 0.0

        # Confidence - Is the confidence level appropriate?
        confidence = explanation.get('confidence_score', 0)
        if 0.5 <= confidence <= 0.95:  # Reasonable confidence range
            scores['confidence_appropriateness'] = 1.0
        else:
            scores['confidence_appropriateness'] = 0.5

        # Overall quality score
        scores['overall'] = sum(scores.values()) / len(scores)

        return scores


# Standalone function for backward compatibility
def generate_dialectical_explanation(alert_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Legacy function for generating dialectical explanations.

    Args:
        alert_data: Alert data to explain

    Returns:
        List of explanation steps (simplified format)
    """
    explainer = DialecticalExplanation()

    # Create a minimal integration result from alert data
    integration_result = {
        'incident_id': alert_data.get('id', 'unknown'),
        'classification': {
            'type': alert_data.get('type', 'unknown'),
            'confidence': 0.8
        },
        'identified_threats': [],
        'recommended_actions': []
    }

    # Generate full explanation
    explanation = explainer.generate_dialectical_explanation(alert_data, integration_result)

    # Convert to simplified format for backward compatibility
    if 'dialectical_process' in explanation:
        return explanation['dialectical_process']['steps']
    else:
        return []