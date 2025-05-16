# src/integration/neurosymbolic_integration.py

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class NeurosymbolicIntegration:
    """
    Integration component that connects neural and symbolic components in A²IR.

    This implements the integration function I in S(KG, R, I) within the neurosymbolic
    pathway: A²IR = Nt[S(KG, R, I)(Ni)]

    The integration layer handles the flow of information between components:
    1. Takes output from neural triage (classifier)
    2. Queries knowledge graph for relevant patterns
    3. Passes combined information to symbolic reasoning
    4. Triggers neural investigation for additional context if needed
    5. Produces final integrated results with explanation traces
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the integration component.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_investigation_rounds = self.config.get('max_investigation_rounds', 3)
        self.debug_mode = self.config.get('debug_mode', False)

        # Initialize tracking for integration process
        self.integration_trace = []
        self.current_incident_id = None

        logger.info("Initialized Neurosymbolic Integration component")

    def integrate(self,
                  alert_data: Dict[str, Any],
                  classifier,
                  knowledge_graph,
                  reasoner,
                  investigator=None) -> Dict[str, Any]:
        """
        Execute the neurosymbolic integration process for an alert.

        Args:
            alert_data: Raw alert data dictionary
            classifier: Neural alert classifier instance
            knowledge_graph: FTAG knowledge graph instance
            reasoner: Symbolic reasoning engine instance
            investigator: Optional neural investigator for context gathering

        Returns:
            Complete integration results with classification, actions, and explanation
        """
        # Reset integration trace for this alert
        self.integration_trace = []
        self.current_incident_id = alert_data.get('id', str(datetime.now().timestamp()))

        # Step 1: Neural Triage (Nt) - Initial Classification
        self._trace_step("Neural Triage", "Starting initial classification")
        classification_result = classifier.classify(alert_data)

        # Add classification info to trace
        self._trace_step(
            "Neural Triage",
            f"Classification: {classification_result.get('classification')} with confidence {classification_result.get('confidence', 0):.2f}"
        )

        # Step 2: Knowledge Graph Query - Find Relevant Patterns
        alert_type = classification_result.get('classification')
        confidence = classification_result.get('confidence', 0)

        # Build entity list from alert data
        entities = self._extract_entities(alert_data)

        # Query knowledge graph with different strategies based on confidence
        if confidence >= self.confidence_threshold:
            self._trace_step(
                "Knowledge Graph",
                f"Querying with high confidence path for {alert_type}"
            )
            kg_query_result = self._query_knowledge_graph(
                knowledge_graph,
                alert_type,
                entities,
                confidence
            )
        else:
            self._trace_step(
                "Knowledge Graph",
                f"Querying with low confidence path for {alert_type}"
            )
            # Use broader query with lower threshold for low confidence
            kg_query_result = self._query_knowledge_graph(
                knowledge_graph,
                alert_type,
                entities,
                max(0.4, confidence - 0.2)  # Lower threshold but not too low
            )

        investigation_rounds = 0
        requires_investigation = False

        # Initial context from alert data
        context = {
            "alert": alert_data,
            "classification": classification_result,
            "kg_data": kg_query_result
        }

        # Step 3: First Symbolic Reasoning
        self._trace_step("Symbolic Reasoning", "Performing initial reasoning")
        reasoning_result = reasoner.reason(classification_result, kg_query_result)

        # Check if investigation is needed and available
        if investigator and self._needs_investigation(reasoning_result):
            requires_investigation = True

        # Step 4: Investigation Loop (if needed)
        while requires_investigation and investigation_rounds < self.max_investigation_rounds:
            investigation_rounds += 1

            self._trace_step(
                "Neural Investigation",
                f"Starting investigation round {investigation_rounds}"
            )

            # Determine what to investigate based on reasoning results
            investigation_targets = self._determine_investigation_targets(reasoning_result)

            # Perform investigation
            investigation_result = self._perform_investigation(
                investigator,
                investigation_targets,
                context
            )

            # Update context with new information
            context["investigation"] = investigation_result

            # Combine with existing knowledge graph data
            kg_query_result = self._update_kg_with_investigation(
                knowledge_graph,
                kg_query_result,
                investigation_result
            )

            # Re-run symbolic reasoning with updated data
            self._trace_step(
                "Symbolic Reasoning",
                f"Re-running reasoning with investigation data (round {investigation_rounds})"
            )

            reasoning_result = reasoner.reason(
                self._combine_neural_outputs(classification_result, investigation_result),
                kg_query_result
            )

            # Check if we need more investigation
            requires_investigation = investigation_rounds < self.max_investigation_rounds and \
                                     self._needs_investigation(reasoning_result)

            if not requires_investigation:
                self._trace_step(
                    "Integration",
                    f"Investigation complete after {investigation_rounds} rounds"
                )

        # Step 5: Generate Final Integration Result
        final_result = self._generate_final_result(
            alert_data,
            classification_result,
            kg_query_result,
            reasoning_result,
            context.get("investigation"),
            self.integration_trace
        )

        self._trace_step("Integration", "Integration process complete")

        return final_result

    def _trace_step(self, component: str, message: str, data: Any = None):
        """
        Record a step in the integration process for explanation and debugging.

        Args:
            component: Component name (Neural, Symbolic, etc.)
            message: Description of the step
            data: Optional data to store with the trace
        """
        timestamp = datetime.now().timestamp()
        trace_entry = {
            "timestamp": timestamp,
            "component": component,
            "message": message
        }

        if data and self.debug_mode:
            trace_entry["data"] = data

        self.integration_trace.append(trace_entry)
        logger.debug(f"[{component}] {message}")

    def _extract_entities(self, alert_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract entities from alert data for knowledge graph queries.

        Args:
            alert_data: Raw alert data

        Returns:
            Dictionary of entity types and values
        """
        entities = {}

        # Extract IPs
        if 'source_ip' in alert_data:
            entities['ip'] = entities.get('ip', []) + [alert_data['source_ip']]

        if 'dest_ip' in alert_data:
            entities['ip'] = entities.get('ip', []) + [alert_data['dest_ip']]

        # Extract domains
        if 'domains' in alert_data:
            entities['domain'] = alert_data['domains']

        # Extract URLs
        if 'urls' in alert_data:
            entities['url'] = alert_data['urls']

        # Extract email addresses
        if 'email' in alert_data and 'sender' in alert_data['email']:
            entities['email'] = entities.get('email', []) + [alert_data['email']['sender']]

        if 'email' in alert_data and 'recipients' in alert_data['email']:
            entities['email'] = entities.get('email', []) + alert_data['email']['recipients']

        # Extract file hashes
        if 'file_hashes' in alert_data:
            entities['hash'] = alert_data['file_hashes']

        # Extract usernames
        if 'username' in alert_data:
            entities['user'] = [alert_data['username']]

        return entities

    def _query_knowledge_graph(self,
                               knowledge_graph,
                               alert_type: str,
                               entities: Dict[str, List[str]],
                               min_confidence: float) -> Dict[str, Any]:
        """
        Query the knowledge graph for attack patterns and threat intelligence.

        Args:
            knowledge_graph: FTAG instance
            alert_type: Type of alert
            entities: Extracted entities
            min_confidence: Minimum confidence threshold

        Returns:
            Knowledge graph query results
        """
        kg_results = {
            "attack_patterns": [],
            "known_threats": [],
            "related_entities": {}
        }

        # NOTE: This is a simplified implementation. In the real A²IR, this would
        # involve complex graph queries. For now, we'll simulate the results.

        # Get attack patterns for the alert type
        # In a real implementation, this would use knowledge_graph.find_attack_paths()
        if hasattr(knowledge_graph, 'find_attack_paths'):
            # Try to find attack paths for each entity
            for entity_type, entity_values in entities.items():
                for entity_value in entity_values:
                    # Convert entity to node ID (this is a simplification)
                    entity_node = f"{entity_type}_{entity_value}"

                    # Look for attack paths
                    if hasattr(knowledge_graph.graph, 'nodes') and entity_node in knowledge_graph.graph.nodes:
                        paths = knowledge_graph.find_attack_paths(
                            entity_node,
                            min_confidence=min_confidence
                        )

                        # Process found paths into attack patterns
                        for path in paths:
                            if path:
                                # Calculate overall path confidence
                                path_confidence = knowledge_graph.get_path_confidence(path) if hasattr(knowledge_graph,
                                                                                                       'get_path_confidence') else min_confidence

                                # Extracting node data would be more complex in a real implementation
                                # This is simplified
                                pattern = {
                                    "id": f"pattern_{len(kg_results['attack_patterns'])}",
                                    "type": alert_type,
                                    "confidence": path_confidence,
                                    "steps": [{"id": f"{source}_{target}", "confidence": conf}
                                              for source, target, conf in path]
                                }
                                kg_results["attack_patterns"].append(pattern)

        return kg_results

    def _needs_investigation(self, reasoning_result: Dict[str, Any]) -> bool:
        """
        Determine if additional investigation is needed based on reasoning results.

        Args:
            reasoning_result: Result from symbolic reasoning

        Returns:
            True if investigation is needed, False otherwise
        """
        # Check if reasoning was successful
        if not reasoning_result.get('success', False):
            return True

        # Check if classifications are confident enough
        classifications = reasoning_result.get('classifications', [])
        if not classifications or classifications[0].get('confidence', 0) < self.confidence_threshold:
            return True

        # Check if there are identified threats
        if not reasoning_result.get('identified_threats', []):
            return True

        # Check if there are recommended actions
        if not reasoning_result.get('actions', []):
            return True

        return False

    def _determine_investigation_targets(self, reasoning_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Determine what to investigate based on reasoning results.

        Args:
            reasoning_result: Result from symbolic reasoning

        Returns:
            List of investigation targets
        """
        targets = []

        # Check for low confidence classifications
        for classification in reasoning_result.get('classifications', []):
            if classification.get('confidence', 0) < self.confidence_threshold:
                targets.append({
                    "type": "classification",
                    "value": classification.get('type'),
                    "reason": "low_confidence"
                })

        # Check for potential threats that need verification
        for threat in reasoning_result.get('potential_threats', []):
            targets.append({
                "type": "threat",
                "value": threat.get('id'),
                "reason": "verification_needed"
            })

        # If no specific targets, add a general investigation
        if not targets:
            targets.append({
                "type": "general",
                "value": "context",
                "reason": "missing_information"
            })

        return targets

    def _perform_investigation(self,
                               investigator,
                               investigation_targets: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform investigation using the neural investigator.

        Args:
            investigator: Neural investigator instance
            investigation_targets: What to investigate
            context: Current context information

        Returns:
            Investigation results
        """
        # If investigator is None, return empty results
        if investigator is None:
            self._trace_step(
                "Neural Investigation",
                "Skipped investigation as no investigator is available"
            )
            return {
                "features": {},
                "entities": {}
            }

        # In a real implementation, this would call the investigator
        # For now, we'll simulate investigation results
        investigation_results = {
            "features": {},
            "entities": {}
        }

        for target in investigation_targets:
            target_type = target.get('type')
            target_value = target.get('value')

            self._trace_step(
                "Neural Investigation",
                f"Investigating {target_type}: {target_value}"
            )

            # Handle based on target type
            if target_type == "classification":
                # Investigate to improve classification confidence
                investigation_results["features"][f"investigated_{target_value}"] = True

            elif target_type == "threat":
                # Investigate specific threat
                investigation_results["entities"]["investigated_threat"] = [target_value]

            elif target_type == "general":
                # General context gathering
                investigation_results["features"]["additional_context_gathered"] = True

        return investigation_results

    def _update_kg_with_investigation(self,
                                      knowledge_graph,
                                      kg_data: Dict[str, Any],
                                      investigation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update knowledge graph data with investigation results.

        Args:
            knowledge_graph: FTAG instance
            kg_data: Current knowledge graph data
            investigation_result: Results from investigation

        Returns:
            Updated knowledge graph data
        """
        # In a real implementation, this would update the FTAG with new information
        # and requery it. For now, we'll just combine the data.

        updated_kg_data = dict(kg_data)

        # Add investigated entities as known threats (simplified)
        if 'entities' in investigation_result:
            for entity_type, entities in investigation_result['entities'].items():
                for entity in entities:
                    new_threat = {
                        "id": f"investigated_threat_{len(updated_kg_data['known_threats'])}",
                        "type": entity_type,
                        "confidence": 0.8,  # Higher confidence after investigation
                        "indicators": [
                            {"type": entity_type, "value": entity}
                        ]
                    }
                    updated_kg_data["known_threats"].append(new_threat)

        return updated_kg_data

    def _combine_neural_outputs(self,
                                classification_result: Dict[str, Any],
                                investigation_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine outputs from neural components.

        Args:
            classification_result: Result from classifier
            investigation_result: Result from investigator

        Returns:
            Combined neural output
        """
        combined = dict(classification_result)

        # Add features from investigation
        if 'features' not in combined:
            combined['features'] = {}

        if 'features' in investigation_result:
            for feature, value in investigation_result['features'].items():
                combined['features'][feature] = value

        # Add entities from investigation
        if 'entities' not in combined:
            combined['entities'] = {}

        if 'entities' in investigation_result:
            for entity_type, entities in investigation_result['entities'].items():
                combined['entities'][entity_type] = entities

        return combined

    def _generate_final_result(self,
                               alert_data: Dict[str, Any],
                               classification_result: Dict[str, Any],
                               kg_data: Dict[str, Any],
                               reasoning_result: Dict[str, Any],
                               investigation_result: Optional[Dict[str, Any]],
                               integration_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate the final integration result.

        Args:
            alert_data: Original alert data
            classification_result: Result from classifier
            kg_data: Knowledge graph data
            reasoning_result: Result from reasoning engine
            investigation_result: Result from investigator (if any)
            integration_trace: Trace of integration steps

        Returns:
            Final integration result
        """
        # Create the final result structure
        final_result = {
            "incident_id": self.current_incident_id,
            "timestamp": datetime.now().timestamp(),
            "original_alert": alert_data,
            "classification": {
                "type": classification_result.get('classification'),
                "confidence": classification_result.get('confidence', 0)
            },
            "identified_threats": reasoning_result.get('identified_threats', []),
            "recommended_actions": reasoning_result.get('actions', []),
            "explanation": {
                "reasoning_path": reasoning_result.get('reasoning_path', []),
                "integration_trace": integration_trace
            }
        }

        return final_result