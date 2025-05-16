# src/symbolic_components/reasoning_engine/asp_reasoner.py

import logging
import os
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set
import json

logger = logging.getLogger(__name__)


class ASPReasoner:
    """
    Symbolic reasoning component for A²IR using Answer Set Programming (ASP).

    This implements the S(KG, R, I) component in the neurosymbolic integration pathway:
    A²IR = Nt[S(KG, R, I)(Ni)]

    The reasoner uses clingo (the ASP solver) to perform logical reasoning over
    the knowledge graph based on outputs from neural components.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ASP reasoning engine.

        Args:
            config: Configuration for the reasoning engine
        """
        self.config = config or {}
        self.rules_path = self.config.get('rules_path', 'data/knowledge_base/asp_rules.lp')
        self.clingo_path = self.config.get('clingo_path', 'clingo')  # Assuming clingo is in PATH
        self.timeout = self.config.get('timeout', 30)  # Timeout in seconds

        # Default confidence threshold for facts
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

        # Load rule sets
        self.rule_sets = {}
        self._load_rule_sets()

        logger.info(f"Initialized ASP reasoner with {sum(len(rules) for rules in self.rule_sets.values())} rules")

    def _load_rule_sets(self):
        """Load ASP rule sets from files."""
        rule_files = {
            'core': 'core_rules.lp',
            'phishing': 'phishing_rules.lp',
            'malware': 'malware_rules.lp',
            'ddos': 'ddos_rules.lp',
            'insider_threat': 'insider_threat_rules.lp'
        }

        for rule_set, file_name in rule_files.items():
            full_path = os.path.join(os.path.dirname(self.rules_path), file_name)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        self.rule_sets[rule_set] = f.read().splitlines()
                    logger.info(f"Loaded {len(self.rule_sets[rule_set])} rules for {rule_set} from {full_path}")
                except Exception as e:
                    logger.error(f"Error loading rule set {rule_set} from {full_path}: {e}")
                    self.rule_sets[rule_set] = []
            else:
                logger.warning(f"Rule file not found: {full_path}")
                self.rule_sets[rule_set] = []

    def _translate_neural_output_to_facts(self, neural_output: Dict[str, Any]) -> List[str]:
        """
        Translate neural component output to ASP facts.

        Args:
            neural_output: Output from neural component(s)

        Returns:
            List of ASP facts
        """
        facts = []

        # Process classification result
        if 'classification' in neural_output and 'confidence' in neural_output:
            classification = neural_output['classification']
            confidence = neural_output['confidence']

            if confidence >= self.confidence_threshold:
                facts.append(f"alert_type({classification}, {confidence:.2f}).")
            else:
                facts.append(f"possible_alert_type({classification}, {confidence:.2f}).")

        # Process features
        if 'features' in neural_output:
            features = neural_output['features']
            for feature_name, feature_value in features.items():
                if isinstance(feature_value, bool):
                    # Boolean features
                    if feature_value:
                        facts.append(f"has_feature({feature_name}).")
                elif isinstance(feature_value, (int, float)):
                    # Numeric features
                    facts.append(f"feature_value({feature_name}, {feature_value}).")
                elif isinstance(feature_value, str):
                    # String features - escape for ASP
                    escaped_value = feature_value.replace("\"", "\\\"")
                    facts.append(f"feature_text({feature_name}, \"{escaped_value}\").")

        # Process entities (like IPs, URLs, etc.)
        if 'entities' in neural_output:
            entities = neural_output['entities']
            for entity_type, entity_values in entities.items():
                for entity_value in entity_values:
                    if isinstance(entity_value, dict) and 'value' in entity_value and 'confidence' in entity_value:
                        # Entity with confidence
                        value = entity_value['value'].replace("\"", "\\\"")
                        conf = entity_value['confidence']
                        facts.append(f"entity({entity_type}, \"{value}\", {conf:.2f}).")
                    else:
                        # Simple entity
                        value = str(entity_value).replace("\"", "\\\"")
                        facts.append(f"entity({entity_type}, \"{value}\").")

        return facts

    def _translate_kg_to_facts(self, kg_data: Dict[str, Any]) -> List[str]:
        """
        Translate knowledge graph data to ASP facts.

        Args:
            kg_data: Data from knowledge graph

        Returns:
            List of ASP facts
        """
        facts = []

        # Process attack patterns
        if 'attack_patterns' in kg_data:
            patterns = kg_data['attack_patterns']
            for pattern in patterns:
                pattern_id = pattern.get('id', '')
                confidence = pattern.get('confidence', 0.0)
                facts.append(f"attack_pattern(\"{pattern_id}\", {confidence:.2f}).")

                # Add pattern steps
                if 'steps' in pattern:
                    for i, step in enumerate(pattern['steps']):
                        step_id = step.get('id', f"{pattern_id}_step{i}")
                        facts.append(f"pattern_step(\"{pattern_id}\", \"{step_id}\", {i}).")

                        # Add step indicators
                        if 'indicators' in step:
                            for indicator in step['indicators']:
                                ind_type = indicator.get('type', '')
                                ind_value = indicator.get('value', '').replace("\"", "\\\"")
                                facts.append(f"step_indicator(\"{step_id}\", \"{ind_type}\", \"{ind_value}\").")

        # Process known threats
        if 'known_threats' in kg_data:
            threats = kg_data['known_threats']
            for threat in threats:
                threat_id = threat.get('id', '')
                threat_type = threat.get('type', '')
                confidence = threat.get('confidence', 0.0)
                facts.append(f"known_threat(\"{threat_id}\", \"{threat_type}\", {confidence:.2f}).")

                # Add threat indicators
                if 'indicators' in threat:
                    for indicator in threat['indicators']:
                        ind_type = indicator.get('type', '')
                        ind_value = indicator.get('value', '').replace("\"", "\\\"")
                        facts.append(f"threat_indicator(\"{threat_id}\", \"{ind_type}\", \"{ind_value}\").")

        return facts

    def _run_clingo(self, facts: List[str], rules: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Run the ASP solver (clingo) with given facts and rules.

        Args:
            facts: List of ASP facts
            rules: List of ASP rules

        Returns:
            Tuple of (success, answer_sets)
        """
        # Create temporary files for facts and combined program
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as facts_file:
            facts_file.write('\n'.join(facts))
            facts_path = facts_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as program_file:
            program_file.write('\n'.join(rules))
            program_path = program_file.name

        try:
            # Run clingo with JSON output
            cmd = [
                self.clingo_path,
                facts_path,
                program_path,
                '--outf=2',  # JSON output
                f'--time-limit={self.timeout}',
                '0'  # Compute all answer sets
            ]

            logger.debug(f"Running clingo with {len(facts)} facts and {len(rules)} rules")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if result.returncode not in [0, 10, 20, 30]:  # Normal, SAT, UNSAT, or UNKNOWN
                logger.error(f"Clingo execution failed with code {result.returncode}: {result.stderr}")
                return False, []

            # Parse JSON output
            try:
                output = json.loads(result.stdout)
                answer_sets = []

                if 'Call' in output and output['Call']:
                    for call in output['Call']:
                        if 'Witnesses' in call:
                            for witness in call['Witnesses']:
                                answer_set = {}
                                for atom in witness.get('Value', []):
                                    # Parse functional-style predicates like p(a,b) into structured data
                                    if '(' in atom and ')' in atom:
                                        pred_name = atom[:atom.find('(')]
                                        pred_args = atom[atom.find('(') + 1:atom.find(')')].split(',')
                                        pred_args = [arg.strip() for arg in pred_args]

                                        # Handle quotes in arguments
                                        parsed_args = []
                                        for arg in pred_args:
                                            if arg.startswith('"') and arg.endswith('"'):
                                                parsed_args.append(arg[1:-1])  # Remove quotes
                                            else:
                                                try:
                                                    # Try to convert to number if possible
                                                    parsed_args.append(float(arg) if '.' in arg else int(arg))
                                                except ValueError:
                                                    parsed_args.append(arg)

                                        # Add to answer set
                                        if pred_name not in answer_set:
                                            answer_set[pred_name] = []
                                        answer_set[pred_name].append(parsed_args)
                                    else:
                                        # Simple predicates like p
                                        if 'simple_predicates' not in answer_set:
                                            answer_set['simple_predicates'] = []
                                        answer_set['simple_predicates'].append(atom)

                                answer_sets.append(answer_set)

                logger.info(f"Clingo found {len(answer_sets)} answer sets")
                return True, answer_sets

            except json.JSONDecodeError:
                logger.error(f"Failed to parse clingo JSON output: {result.stdout}")
                return False, []

        except Exception as e:
            logger.error(f"Error running clingo: {e}", exc_info=True)
            return False, []
        finally:
            # Clean up temporary files
            try:
                os.unlink(facts_path)
                os.unlink(program_path)
            except Exception as cleanup_error:
                logger.warning(f"Error removing temporary files: {cleanup_error}")

    def reason(self, neural_output: Dict[str, Any], kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform symbolic reasoning based on neural outputs and knowledge graph data.

        Args:
            neural_output: Output from neural components
            kg_data: Data from knowledge graph

        Returns:
            Reasoning results including classifications, identified threats, and recommended actions
        """
        # Translate inputs to ASP facts
        neural_facts = self._translate_neural_output_to_facts(neural_output)
        kg_facts = self._translate_kg_to_facts(kg_data)
        all_facts = neural_facts + kg_facts

        # Determine which rule sets to use based on classification
        rule_sets_to_use = ['core']  # Always use core rules

        # Add alert-type specific rules if available
        alert_type = neural_output.get('classification')
        if alert_type and alert_type in self.rule_sets:
            rule_sets_to_use.append(alert_type)

        # Combine rules from selected sets
        all_rules = []
        for rule_set in rule_sets_to_use:
            all_rules.extend(self.rule_sets.get(rule_set, []))

        # Run ASP solver
        success, answer_sets = self._run_clingo(all_facts, all_rules)

        if not success or not answer_sets:
            logger.warning("Reasoning produced no results")
            return {
                "success": False,
                "classifications": [],
                "identified_threats": [],
                "actions": []
            }

        # Process answer sets (usually take the first/best one)
        answer_set = answer_sets[0]  # Take first answer set

        # Extract classifications with confidence
        classifications = []
        for classification in answer_set.get('alert_classification', []):
            if len(classification) >= 2:
                classifications.append({
                    "type": classification[0],
                    "confidence": classification[1]
                })

        # Extract identified threats
        threats = []
        for threat in answer_set.get('identified_threat', []):
            if len(threat) >= 3:
                threats.append({
                    "id": threat[0],
                    "type": threat[1],
                    "confidence": threat[2]
                })

        # Extract recommended actions
        actions = []
        for action in answer_set.get('recommended_action', []):
            if len(action) >= 3:
                actions.append({
                    "action": action[0],
                    "target": action[1],
                    "priority": action[2]
                })

        # Extract reasoning path for explanation
        reasoning_path = []
        for rule in answer_set.get('used_rule', []):
            if len(rule) >= 1:
                reasoning_path.append({
                    "rule_id": rule[0],
                    "description": rule[1] if len(rule) > 1 else ""
                })

        result = {
            "success": True,
            "classifications": classifications,
            "identified_threats": threats,
            "actions": actions,
            "reasoning_path": reasoning_path
        }

        return result

    def generate_explanation(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a structured explanation from reasoning results.

        Args:
            reasoning_result: Results from the reason() method

        Returns:
            Structured explanation for the dialectical explanation framework
        """
        explanation = {
            "alert_type": None,
            "confidence": 0.0,
            "threats": [],
            "reasoning_chain": [],
            "recommended_actions": []
        }

        # Set alert type and confidence
        if reasoning_result['classifications']:
            explanation['alert_type'] = reasoning_result['classifications'][0]['type']
            explanation['confidence'] = reasoning_result['classifications'][0]['confidence']

        # Add identified threats
        explanation['threats'] = reasoning_result['identified_threats']

        # Add reasoning chain
        explanation['reasoning_chain'] = reasoning_result['reasoning_path']

        # Add recommended actions
        explanation['recommended_actions'] = reasoning_result['actions']

        return explanation