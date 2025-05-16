# src/evaluation/framework_evaluator.py

import logging
import os
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from .metrics_calculator import MetricsCalculator
from .report_generator import ReportGenerator
from src.visualization.evaluation_visualizer import EvaluationVisualizer
from src.agents.coordination.incident_workflow import IncidentWorkflow

logger = logging.getLogger(__name__)


class FrameworkEvaluator:
    """
    Main coordinator for A²IR framework evaluation.
    Delegates specific tasks to specialized components.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the framework evaluator."""
        self.config = config
        self.evaluation_config = config.get('evaluation', {})

        # Output directory
        self.output_dir = self.evaluation_config.get('output_dir', 'data/evaluation')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize components
        self.metrics_calculator = MetricsCalculator(config)
        self.report_generator = ReportGenerator(config)
        self.visualizer = EvaluationVisualizer(config)

        logger.info("Framework evaluator initialized")

    def evaluate_framework(self,
                           test_data_path: str,
                           ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the A²IR framework.

        Args:
            test_data_path: Path to test data
            ground_truth_path: Optional path to ground truth labels

        Returns:
            Evaluation report
        """
        logger.info(f"Starting framework evaluation with test data: {test_data_path}")

        start_time = time.time()

        try:
            # Initialize workflow
            workflow = IncidentWorkflow(self.config)
            if not workflow.initialize_system():
                raise RuntimeError("Failed to initialize workflow")

            # Load test data
            test_data = self._load_test_data(test_data_path)
            ground_truth = self._load_ground_truth(ground_truth_path) if ground_truth_path else {}

            # Process incidents
            processing_results = self._process_incidents(workflow, test_data, ground_truth)

            # Calculate metrics
            evaluation_metrics = self.metrics_calculator.calculate_all_metrics(processing_results)

            # Generate visualizations if enabled
            if self.evaluation_config.get('generate_plots', True):
                self.visualizer.create_all_visualizations(evaluation_metrics, processing_results)

            # Generate report
            report = self.report_generator.generate_full_report(
                evaluation_metrics,
                processing_results,
                time.time() - start_time
            )

            # Save results
            self._save_results(report)

            logger.info("Framework evaluation complete")
            return report

        except Exception as e:
            logger.error(f"Error during framework evaluation: {e}", exc_info=True)
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }

    def _process_incidents(self,
                           workflow: IncidentWorkflow,
                           test_data: List[Dict[str, Any]],
                           ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process all test incidents."""
        processing_results = []

        for i, incident in enumerate(test_data):
            logger.info(f"Processing incident {i + 1}/{len(test_data)}")

            # Time the processing
            incident_start = time.time()
            result = workflow.process_single_incident(incident)
            incident_time = time.time() - incident_start

            # Add timing and ground truth
            result['actual_processing_time'] = incident_time

            incident_id = incident.get('id', str(i))
            if incident_id in ground_truth:
                result['ground_truth'] = ground_truth[incident_id]

            processing_results.append(result)

        return processing_results

    def _load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """Load test data from file."""
        import json
        import pandas as pd

        if test_data_path.endswith('.json'):
            with open(test_data_path, 'r') as f:
                data = json.load(f)
        elif test_data_path.endswith('.csv'):
            df = pd.read_csv(test_data_path)
            data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {test_data_path}")

        return data if isinstance(data, list) else [data]

    def _load_ground_truth(self, ground_truth_path: str) -> Dict[str, Any]:
        """Load ground truth labels."""
        import json

        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        # Convert to dictionary keyed by incident ID
        if isinstance(ground_truth, list):
            return {item.get('id', str(i)): item for i, item in enumerate(ground_truth)}
        return ground_truth

    def _save_results(self, report: Dict[str, Any]):
        """Save evaluation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save full report as JSON
        report_file = os.path.join(
            self.output_dir,
            f'evaluation_report_{timestamp}.json'
        )
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Save summary as markdown
        summary_file = os.path.join(
            self.output_dir,
            f'evaluation_summary_{timestamp}.md'
        )
        with open(summary_file, 'w') as f:
            f.write(report['summary'])

        logger.info(f"Evaluation results saved to {self.output_dir}")


def run_evaluation(config: Dict[str, Any], test_data_path: str, ground_truth_path: Optional[str] = None):
    """Convenience function to run framework evaluation."""
    evaluator = FrameworkEvaluator(config)
    return evaluator.evaluate_framework(test_data_path, ground_truth_path)