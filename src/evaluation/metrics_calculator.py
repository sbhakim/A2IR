# src/evaluation/metrics_calculator.py

import logging
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.trust_framework.trust_score import TrustScore

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates various metrics for A²IR framework evaluation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics calculator."""
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        self.target_values = self.evaluation_config.get('target_values', {})

        # Initialize trust calculator
        self.trust_calculator = TrustScore(config.get('trust_score', {}))

    def calculate_all_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all evaluation metrics."""
        metrics = {
            'classification_metrics': self.calculate_classification_metrics(results),
            'performance_metrics': self.calculate_performance_metrics(results),
            'explanation_metrics': self.calculate_explanation_metrics(results),
            'trust_metrics': self.calculate_trust_metrics(results),
            'knowledge_graph_metrics': self.calculate_kg_metrics(results),
            'overall_metrics': {}
        }

        # Calculate overall metrics
        metrics['overall_metrics'] = self.calculate_overall_metrics(metrics)

        return metrics

    def calculate_classification_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate classification accuracy metrics."""
        true_labels = []
        predicted_labels = []
        confidence_scores = []

        for result in results:
            if 'ground_truth' in result and 'classification' in result:
                true_labels.append(result['ground_truth'].get('type', 'unknown'))
                predicted_labels.append(result['classification'].get('type', 'unknown'))
                confidence_scores.append(result['classification'].get('confidence', 0))

        if not true_labels:
            return {'error': 'No ground truth available for classification metrics'}

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='weighted'
        )

        classification_report_dict = classification_report(
            true_labels, predicted_labels, output_dict=True
        )

        return {
            'accuracy': sum(p == t for p, t in zip(predicted_labels, true_labels)) / len(true_labels),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report_dict,
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            }
        }

    def calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        processing_times = []
        successful_processing = 0

        for result in results:
            if 'actual_processing_time' in result:
                processing_times.append(result['actual_processing_time'])

            if not result.get('error'):
                successful_processing += 1

        if not processing_times:
            return {'error': 'No processing time data available'}

        # Calculate baseline (simulated manual processing time)
        baseline_time = 300  # 5 minutes per incident

        return {
            'processing_times': {
                'mean': np.mean(processing_times),
                'std': np.std(processing_times),
                'min': np.min(processing_times),
                'max': np.max(processing_times),
                'median': np.median(processing_times)
            },
            'success_rate': successful_processing / len(results),
            'throughput': len(results) / sum(processing_times) if processing_times else 0,
            'response_time_reduction': 1 - (np.mean(processing_times) / baseline_time),
            'total_incidents': len(results),
            'successful_incidents': successful_processing
        }

    def calculate_explanation_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate explanation quality metrics."""
        explanation_scores = []
        completeness_scores = []

        for result in results:
            if 'explanation' in result:
                explanation = result['explanation']

                # Check completeness
                required_elements = ['reasoning_path', 'integration_trace']
                present_elements = sum(1 for elem in required_elements if elem in explanation)
                completeness = present_elements / len(required_elements)
                completeness_scores.append(completeness)

                # Check explanation depth
                reasoning_steps = len(explanation.get('reasoning_path', []))
                integration_steps = len(explanation.get('integration_trace', []))

                explanation_scores.append({
                    'reasoning_steps': reasoning_steps,
                    'integration_steps': integration_steps,
                    'total_steps': reasoning_steps + integration_steps
                })

        if not explanation_scores:
            return {'error': 'No explanation data available'}

        return {
            'completeness': {
                'mean': np.mean(completeness_scores),
                'std': np.std(completeness_scores)
            },
            'depth': {
                'mean_reasoning_steps': np.mean([s['reasoning_steps'] for s in explanation_scores]),
                'mean_integration_steps': np.mean([s['integration_steps'] for s in explanation_scores]),
                'mean_total_steps': np.mean([s['total_steps'] for s in explanation_scores])
            }
        }

    def calculate_trust_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trust score metrics."""
        trust_scores = []
        trust_components = []

        for result in results:
            if 'explanation' in result and 'classification' in result:
                # Calculate comprehensive trust score
                trust_evaluation = self.trust_calculator.calculate_comprehensive_trust_score(
                    result['explanation'],
                    result['classification']
                )

                trust_scores.append(trust_evaluation['trust_score'])
                trust_components.append(trust_evaluation)

        if not trust_scores:
            return {'error': 'No trust score data available'}

        # Analyze trust score trends
        trust_analysis = self.trust_calculator.analyze_trust_trends()

        return {
            'trust_scores': {
                'mean': np.mean(trust_scores),
                'std': np.std(trust_scores),
                'min': np.min(trust_scores),
                'max': np.max(trust_scores)
            },
            'cognitive_alignment': self._aggregate_cognitive_scores(trust_components),
            'trust_trends': trust_analysis
        }

    def _aggregate_cognitive_scores(self, trust_components: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate cognitive alignment scores."""
        cognitive_dimensions = ['clarity', 'completeness', 'actionability', 'coherence', 'relevance']
        aggregated = {}

        for dimension in cognitive_dimensions:
            scores = []
            for component in trust_components:
                if 'cognitive_scores' in component and dimension in component['cognitive_scores']:
                    scores.append(component['cognitive_scores'][dimension])

            if scores:
                aggregated[dimension] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }

        return aggregated

    def calculate_kg_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate knowledge graph utilization metrics."""
        kg_hits = 0
        kg_patterns_used = []

        for result in results:
            if 'explanation' in result:
                explanation = result['explanation']
                reasoning_path = explanation.get('reasoning_path', [])

                # Check if knowledge graph was used
                for step in reasoning_path:
                    if 'rule_id' in step or 'pattern' in step:
                        kg_hits += 1
                        if 'pattern' in step:
                            kg_patterns_used.append(step['pattern'])
                        break

        unique_patterns = len(set(kg_patterns_used))

        return {
            'utilization_rate': kg_hits / len(results) if results else 0,
            'unique_patterns_used': unique_patterns,
            'pattern_diversity': unique_patterns / kg_hits if kg_hits > 0 else 0
        }

    def calculate_overall_metrics(self, component_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall framework performance metrics."""
        classification = component_metrics.get('classification_metrics', {})
        performance = component_metrics.get('performance_metrics', {})
        trust = component_metrics.get('trust_metrics', {})

        # Check against target values
        targets_met = {}

        # F1 Score target
        if 'f1_score' in classification:
            f1_target = self.target_values.get('f1_score', 0.85)
            targets_met['f1_score'] = {
                'value': classification['f1_score'],
                'target': f1_target,
                'met': classification['f1_score'] >= f1_target
            }

        # Response time reduction target
        if 'response_time_reduction' in performance:
            time_target = self.target_values.get('response_time_reduction', 0.6)
            targets_met['response_time_reduction'] = {
                'value': performance['response_time_reduction'],
                'target': time_target,
                'met': performance['response_time_reduction'] >= time_target
            }

        # Trust score target
        if 'trust_scores' in trust and 'mean' in trust['trust_scores']:
            trust_target = self.target_values.get('trust_score', 4.0)
            targets_met['trust_score'] = {
                'value': trust['trust_scores']['mean'],
                'target': trust_target,
                'met': trust['trust_scores']['mean'] >= trust_target
            }

        return {
            'targets_met': targets_met,
            'overall_success_rate': sum(1 for t in targets_met.values() if t['met']) / len(
                targets_met) if targets_met else 0,
            'summary': {
                'classification_f1': classification.get('f1_score', 0),
                'response_time_reduction': performance.get('response_time_reduction', 0),
                'average_trust_score': trust.get('trust_scores', {}).get('mean', 0),
                'success_rate': performance.get('success_rate', 0)
            }
        }