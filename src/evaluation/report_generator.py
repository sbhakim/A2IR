# src/evaluation/report_generator.py

import logging
import os
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates evaluation reports and summaries."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the report generator."""
        self.config = config
        self.evaluation_config = config.get('evaluation', {})

    def generate_full_report(self,
                             metrics: Dict[str, Any],
                             results: List[Dict[str, Any]],
                             total_time: float) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluation_duration': total_time,
                'total_incidents': len(results),
                'configuration': self.evaluation_config
            },
            'metrics': metrics,
            'target_achievement': self._assess_target_achievement(metrics),
            'summary': self._generate_summary(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }

        return report

    def _assess_target_achievement(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess achievement of target values."""
        targets_met = metrics.get('overall_metrics', {}).get('targets_met', {})

        achievement = {
            'overall_success': all(t['met'] for t in targets_met.values()) if targets_met else False,
            'detailed_results': targets_met,
            'success_rate': sum(1 for t in targets_met.values() if t['met']) / len(targets_met) if targets_met else 0
        }

        return achievement

    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate executive summary."""
        overall = metrics.get('overall_metrics', {}).get('summary', {})
        targets = metrics.get('overall_metrics', {}).get('targets_met', {})

        summary_parts = [
            "# A²IR Framework Evaluation Summary\n",
            f"## Overall Performance\n",
            f"- Classification F1 Score: {overall.get('classification_f1', 0):.2f}",
            f"- Response Time Reduction: {overall.get('response_time_reduction', 0):.1%}",
            f"- Average Trust Score: {overall.get('average_trust_score', 0):.2f}/5.0",
            f"- Success Rate: {overall.get('success_rate', 0):.1%}\n",
            f"## Target Achievement\n"
        ]

        for metric, result in targets.items():
            status = "✓" if result['met'] else "✗"
            summary_parts.append(
                f"- {metric}: {result['value']:.2f} vs {result['target']:.2f} {status}"
            )

        return "\n".join(summary_parts)

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []

        # Check classification performance
        classification = metrics.get('classification_metrics', {})
        if classification.get('f1_score', 0) < 0.85:
            recommendations.append(
                "Classification performance below target. Consider additional training data or model tuning."
            )

        # Check response time
        performance = metrics.get('performance_metrics', {})
        if performance.get('response_time_reduction', 0) < 0.6:
            recommendations.append(
                "Response time reduction below target. Investigate performance bottlenecks in the pipeline."
            )

        # Check trust scores
        trust = metrics.get('trust_metrics', {})
        if trust.get('trust_scores', {}).get('mean', 0) < 4.0:
            recommendations.append(
                "Trust scores below target. Enhance explanation quality and cognitive alignment."
            )

        # Check cognitive dimensions
        cognitive = trust.get('cognitive_alignment', {})
        for dimension, scores in cognitive.items():
            if scores.get('mean', 0) < 3.5:
                recommendations.append(
                    f"Low {dimension} scores. Focus on improving this aspect of explanations."
                )

        # Check knowledge graph utilization
        kg_metrics = metrics.get('knowledge_graph_metrics', {})
        if kg_metrics.get('utilization_rate', 0) < 0.7:
            recommendations.append(
                "Low knowledge graph utilization. Expand the knowledge base or improve integration."
            )

        if not recommendations:
            recommendations.append("All metrics meet or exceed targets. Continue monitoring performance.")

        return recommendations