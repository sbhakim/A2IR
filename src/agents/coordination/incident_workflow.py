# src/agents/coordination/incident_workflow.py

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import all the components we've built
from src.neural_components.alert_classification.classifier import AlertClassifier
from src.neural_components.context_gathering.neural_investigator import NeuralInvestigator
from src.knowledge_graph.ftags.ftag_construct import FuzzyTemporalAttackGraph
from src.knowledge_graph.ftags.ftag_initializer import initialize_knowledge_graph
from src.symbolic_components.reasoning_engine.asp_reasoner import ASPReasoner
from src.integration.neurosymbolic_integration import NeurosymbolicIntegration
from src.data_processing.alert_preprocessor import AlertPreprocessor
from src.explanation_framework.dialectical_explanation import DialecticalExplanation
from src.trust_framework.trust_score import TrustScore

logger = logging.getLogger(__name__)


class IncidentWorkflow:
    """
    Complete workflow coordinator for processing security incidents in A²IR.

    This class implements the full agent coordination for:
    1. Processing incidents through the complete pipeline
    2. Coordinating between all neural and symbolic components
    3. Producing structured responses with explanations and trust scores
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the incident workflow with all required components.

        Args:
            config: Application configuration
        """
        self.config = config

        # Initialize components
        logger.info("Initializing workflow components...")

        # Neural components
        self.classifier = AlertClassifier(config.get('models', {}).get('alert_classifier', {}))
        self.investigator = NeuralInvestigator(config.get('neural_investigator', {}))

        # Knowledge graph
        kg_config = config.get('knowledge_graph', {})
        self.knowledge_graph = None  # Will be loaded in initialize_system

        # Symbolic reasoning
        self.reasoner = ASPReasoner(config.get('reasoning', {}))

        # Integration layer
        integration_config = config.get('integration', {})
        # Pass investigator explicitly
        integration_config['enable_investigation'] = True
        self.integrator = NeurosymbolicIntegration(integration_config)

        # Explanation and trust frameworks
        self.explainer = DialecticalExplanation(config.get('explanation', {}))
        self.trust_calculator = TrustScore(config.get('trust_score', {}))

        # Preprocessor
        self.preprocessor = AlertPreprocessor(config.get('preprocessing', {}))

        # Workflow state
        self.state = 'initialized'
        self.processed_incidents = []

        # Initialize paths
        self.setup_paths()

    def setup_paths(self):
        """Create necessary directories for models and data."""
        paths_to_create = [
            'models/alert_classifier',
            'models/knowledge_graph',
            'data/processed',
            'data/results',
            'logs',
            'data/knowledge_base'
        ]

        for path in paths_to_create:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")

    def initialize_system(self) -> bool:
        """
        Initialize the system by loading models and knowledge graph.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing A²IR system...")

            # Load alert classifier model if exists
            model_path = self.config.get('models', {}).get('alert_classifier', {}).get('model_path',
                                                                                       'models/alert_classifier')
            if self.classifier.load_model(model_path):
                logger.info("Alert classifier model loaded successfully")
            else:
                logger.warning("No pre-trained classifier model found. You'll need to train one.")

            # Load or initialize knowledge graph
            kg_path = self.config.get('knowledge_graph', {}).get('storage_path', 'models/knowledge_graph/a2ir_kg.json')
            if os.path.exists(kg_path):
                self.knowledge_graph = FuzzyTemporalAttackGraph.load_from_file(kg_path)
                logger.info("Knowledge graph loaded successfully")
            else:
                logger.warning("No existing knowledge graph found. Initializing new one...")
                self.knowledge_graph = initialize_knowledge_graph(self.config)
                if self.knowledge_graph.save_to_file(kg_path):
                    logger.info(f"New knowledge graph saved to {kg_path}")

            # Setup ASP rules if they don't exist
            self._setup_asp_rules()

            self.state = 'ready'
            logger.info("A²IR system initialization complete")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}", exc_info=True)
            self.state = 'error'
            return False

    def _setup_asp_rules(self):
        """Set up ASP rules if they don't exist."""
        kb_dir = "data/knowledge_base"
        os.makedirs(kb_dir, exist_ok=True)

        # Check if rules already exist
        rule_files = ['core_rules.lp', 'phishing_rules.lp', 'malware_rules.lp']
        if all(os.path.exists(os.path.join(kb_dir, f)) for f in rule_files):
            logger.info("ASP rules already exist")
            return

        # Create basic rules if they don't exist
        # ... (same as original implementation)

    def process_single_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single security incident through the complete A²IR pipeline.

        Args:
            incident_data: Incident data (already standardized)

        Returns:
            Complete processing result with classification, actions, explanation, and trust score
        """
        if self.state != 'ready':
            raise RuntimeError("System not ready. Please initialize first.")

        logger.info(f"Processing incident: {incident_data.get('id', 'unknown')}")

        start_time = time.time()

        try:
            # Run the neurosymbolic integration process with investigation
            integration_result = self.integrator.integrate(
                alert_data=incident_data.get('data', incident_data),
                classifier=self.classifier,
                knowledge_graph=self.knowledge_graph,
                reasoner=self.reasoner,
                investigator=self.investigator  # Pass the investigator
            )

            # Generate dialectical explanation
            dialectical_explanation = self.explainer.generate_dialectical_explanation(
                incident_data,
                integration_result
            )

            # Calculate trust score
            trust_evaluation = self.trust_calculator.calculate_comprehensive_trust_score(
                dialectical_explanation,
                integration_result.get('classification', {})
            )

            # Update knowledge graph with new patterns if confidence is high
            if integration_result.get('classification', {}).get('confidence', 0) > 0.8:
                self._update_knowledge_graph(integration_result)

            # Compile final result
            final_result = {
                'incident_id': integration_result.get('incident_id'),
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'classification': integration_result.get('classification'),
                'identified_threats': integration_result.get('identified_threats', []),
                'recommended_actions': integration_result.get('recommended_actions', []),
                'explanation': dialectical_explanation,
                'trust_evaluation': trust_evaluation,
                'integration_trace': integration_result.get('explanation', {}).get('integration_trace', [])
            }

            # Store in processed incidents
            self.processed_incidents.append(final_result)

            logger.info(f"Incident processed successfully in {final_result['processing_time']:.2f} seconds")
            logger.info(f"Trust score: {trust_evaluation['trust_score']:.2f}")

            return final_result

        except Exception as e:
            logger.error(f"Error processing incident: {e}", exc_info=True)
            error_result = {
                "incident_id": incident_data.get('id', 'unknown'),
                "error": str(e),
                "success": False,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            self.processed_incidents.append(error_result)
            return error_result

    def _update_knowledge_graph(self, integration_result: Dict[str, Any]):
        """Update knowledge graph with newly identified patterns."""
        try:
            classification = integration_result.get('classification', {})
            alert_type = classification.get('type')
            confidence = classification.get('confidence', 0)

            if not alert_type or confidence < 0.8:
                return

            # Add new pattern node if it doesn't exist
            pattern_id = f"learned_pattern_{alert_type}_{int(time.time())}"

            self.knowledge_graph.add_node(pattern_id, "learned_pattern", {
                "name": f"Learned {alert_type} pattern",
                "confidence": confidence,
                "source": "incident_processing",
                "timestamp": datetime.now().isoformat()
            })

            # Connect to main pattern if exists
            main_pattern_id = f"{alert_type}_campaign"
            if main_pattern_id in self.knowledge_graph.graph.nodes:
                self.knowledge_graph.add_edge(
                    pattern_id,
                    main_pattern_id,
                    confidence=confidence,
                    temporal_constraint=3600.0,  # 1 hour
                    relation_type="learned_from"
                )

            logger.debug(f"Updated knowledge graph with new pattern: {pattern_id}")

        except Exception as e:
            logger.warning(f"Failed to update knowledge graph: {e}")

    def process_file(self, input_file: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process all incidents in a file.

        Args:
            input_file: Path to file containing incidents
            output_dir: Directory to save results

        Returns:
            List of processing results
        """
        logger.info(f"Processing incidents from file: {input_file}")

        # Preprocess the file
        alerts = self.preprocessor.preprocess_file(input_file)

        if not alerts:
            logger.warning(f"No alerts found in file: {input_file}")
            return []

        logger.info(f"Found {len(alerts)} alerts to process")

        # Process each alert
        results = []
        for i, alert in enumerate(alerts):
            logger.info(f"Processing alert {i + 1}/{len(alerts)}")
            result = self.process_single_incident(alert)
            results.append(result)

            # Optionally save intermediate results to avoid losing progress
            if i % 10 == 0 and output_dir:
                self._save_intermediate_results(results, output_dir)

        # Save final results
        if output_dir:
            self._save_final_results(results, output_dir)

        # Generate summary report
        self._generate_summary_report(results, output_dir)

        return results

    def _save_intermediate_results(self, results: List[Dict[str, Any]], output_dir: str):
        """Save intermediate results during processing."""
        os.makedirs(output_dir, exist_ok=True)
        intermediate_file = os.path.join(output_dir, 'intermediate_results.json')

        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _save_final_results(self, results: List[Dict[str, Any]], output_dir: str):
        """Save final processing results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save full results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'results_{timestamp}.json')

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_file = os.path.join(output_dir, f'summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            self._write_summary(f, results)

        logger.info(f"Results saved to: {output_dir}")

    def _write_summary(self, file_handle, results: List[Dict[str, Any]]):
        """Write a summary of processing results."""
        total = len(results)
        successful = sum(1 for r in results if not r.get('error'))

        file_handle.write(f"A²IR Processing Summary\n")
        file_handle.write(f"======================\n\n")
        file_handle.write(f"Total incidents processed: {total}\n")
        file_handle.write(f"Successful: {successful}\n")
        file_handle.write(f"Errors: {total - successful}\n\n")

        # Alert type distribution
        alert_types = {}
        for r in results:
            if 'classification' in r:
                alert_type = r['classification'].get('type', 'unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

        file_handle.write("Alert Type Distribution:\n")
        for alert_type, count in alert_types.items():
            file_handle.write(f"  {alert_type}: {count}\n")

        # Trust score statistics
        trust_scores = [r['trust_evaluation']['trust_score']
                        for r in results
                        if 'trust_evaluation' in r and 'trust_score' in r['trust_evaluation']]

        if trust_scores:
            file_handle.write(f"\nTrust Score Statistics:\n")
            file_handle.write(f"  Average: {sum(trust_scores) / len(trust_scores):.2f}\n")
            file_handle.write(f"  Min: {min(trust_scores):.2f}\n")
            file_handle.write(f"  Max: {max(trust_scores):.2f}\n")

    def _generate_summary_report(self, results: List[Dict[str, Any]], output_dir: str = None):
        """Generate a detailed summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_incidents': len(results),
            'successful_processing': sum(1 for r in results if not r.get('error')),
            'average_processing_time': sum(r.get('processing_time', 0) for r in results) / len(
                results) if results else 0,
            'alert_type_distribution': {},
            'trust_score_statistics': {},
            'action_recommendations': {}
        }

        # Analyze results
        for result in results:
            # Alert type distribution
            if 'classification' in result:
                alert_type = result['classification'].get('type', 'unknown')
                report['alert_type_distribution'][alert_type] = \
                    report['alert_type_distribution'].get(alert_type, 0) + 1

            # Action recommendations
            if 'recommended_actions' in result:
                for action in result['recommended_actions']:
                    action_type = action.get('action', 'unknown')
                    report['action_recommendations'][action_type] = \
                        report['action_recommendations'].get(action_type, 0) + 1

        # Trust score analysis
        trust_scores = [r['trust_evaluation']['trust_score']
                        for r in results
                        if 'trust_evaluation' in r and 'trust_score' in r['trust_evaluation']]

        if trust_scores:
            report['trust_score_statistics'] = {
                'mean': sum(trust_scores) / len(trust_scores),
                'min': min(trust_scores),
                'max': max(trust_scores),
                'count': len(trust_scores)
            }

        # Save report if output directory provided
        if output_dir:
            report_file = os.path.join(output_dir, f'summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report saved to: {report_file}")

        return report

    def save_knowledge_graph(self):
        """Save the current state of the knowledge graph."""
        kg_path = self.config.get('knowledge_graph', {}).get('storage_path', 'models/knowledge_graph/a2ir_kg.json')
        if self.knowledge_graph.save_to_file(kg_path):
            logger.info(f"Knowledge graph saved to: {kg_path}")
        else:
            logger.error("Failed to save knowledge graph")


def run_workflow(config: Dict[str, Any], mode: str, **kwargs):
    """
    Main entry point for running the incident workflow.

    Args:
        config: Application configuration
        mode: Operation mode (process, train, evaluate)
        **kwargs: Additional arguments based on mode
    """
    workflow = IncidentWorkflow(config)

    # Initialize the system
    if not workflow.initialize_system():
        logger.error("System initialization failed")
        return

    if mode == 'process':
        # Process incidents from a file
        input_file = kwargs.get('input_file')
        output_dir = kwargs.get('output_dir', 'data/results')
        if input_file:
            results = workflow.process_file(input_file, output_dir)
            logger.info(f"Processed {len(results)} incidents")
        else:
            logger.error("No input file specified for processing")

    else:
        logger.error(f"Unknown mode: {mode}")

    # Always save the knowledge graph after operations
    workflow.save_knowledge_graph()