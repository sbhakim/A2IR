# src/visualization/evaluation_visualizer.py

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """Creates visualizations for A²IR evaluation results."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualizer."""
        self.config = config
        self.output_dir = config.get('evaluation', {}).get('output_dir', 'data/evaluation')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        # Set default style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.dpi'] = 300

    def create_all_visualizations(self,
                                  metrics: Dict[str, Any],
                                  results: List[Dict[str, Any]]):
        """Create all evaluation visualizations."""
        logger.info("Creating evaluation visualizations...")

        # 1. Classification performance
        if 'classification_metrics' in metrics:
            self.plot_classification_performance(metrics['classification_metrics'])

        # 2. Response time distribution
        if 'performance_metrics' in metrics:
            self.plot_response_times(metrics['performance_metrics'], results)

        # 3. Trust score distribution
        if 'trust_metrics' in metrics:
            self.plot_trust_scores(metrics['trust_metrics'])

        # 4. Overall performance radar chart
        if 'overall_metrics' in metrics:
            self.plot_performance_radar(metrics['overall_metrics'])

        logger.info(f"Visualizations saved to {self.plots_dir}")

    def plot_classification_performance(self, classification_metrics: Dict[str, Any]):
        """Plot classification performance metrics."""
        report = classification_metrics.get('classification_report', {})

        # Extract per-class metrics
        classes = []
        precisions = []
        recalls = []
        f1_scores = []

        for class_name, class_metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue

            classes.append(class_name)
            precisions.append(class_metrics['precision'])
            recalls.append(class_metrics['recall'])
            f1_scores.append(class_metrics['f1-score'])

        if not classes:
            logger.warning("No classification data to plot")
            return

        # Create grouped bar chart
        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label='Precision')
        ax.bar(x, recalls, width, label='Recall')
        ax.bar(x + width, f1_scores, width, label='F1-Score')

        ax.set_xlabel('Alert Types')
        ax.set_ylabel('Score')
        ax.set_title('Classification Performance by Alert Type')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for i, v in enumerate(f1_scores):
            ax.text(i + width, v + 0.01, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        self._save_plot('classification_performance.png')

    def plot_response_times(self,
                            performance_metrics: Dict[str, Any],
                            results: List[Dict[str, Any]]):
        """Plot response time distribution and comparison."""
        processing_times = [r['actual_processing_time'] for r in results
                            if 'actual_processing_time' in r]

        if not processing_times:
            logger.warning("No processing time data to plot")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Response time distribution
        ax1.hist(processing_times, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(processing_times), color='red', linestyle='--',
                    label=f'Mean: {np.mean(processing_times):.2f}s')
        ax1.axvline(np.median(processing_times), color='green', linestyle='--',
                    label=f'Median: {np.median(processing_times):.2f}s')
        ax1.set_xlabel('Response Time (seconds)')
        ax1.set_ylabel('Count')
        ax1.set_title('Response Time Distribution')
        ax1.legend()

        # Before/After comparison
        baseline_time = 300  # 5 minutes manual processing
        avg_time = np.mean(processing_times)

        categories = ['Manual Process', 'A²IR System']
        times = [baseline_time, avg_time]
        colors = ['red', 'green']

        bars = ax2.bar(categories, times, color=colors, alpha=0.7)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Response Time Comparison')

        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}s', ha='center', va='bottom')

        # Add reduction percentage
        reduction = performance_metrics.get('response_time_reduction', 0)
        ax2.text(0.5, max(times) * 0.5,
                 f'{reduction:.1%} Reduction',
                 ha='center', va='center',
                 transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=14, fontweight='bold')

        plt.tight_layout()
        self._save_plot('response_times.png')

    def plot_trust_scores(self, trust_metrics: Dict[str, Any]):
        """Plot trust score distribution and cognitive dimensions."""
        trust_scores = trust_metrics.get('trust_scores', {})
        cognitive_alignment = trust_metrics.get('cognitive_alignment', {})

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Trust score distribution (simulated)
        if 'mean' in trust_scores and 'std' in trust_scores:
            # Generate sample data based on mean and std
            scores = np.random.normal(
                trust_scores['mean'],
                trust_scores['std'],
                100
            )
            scores = np.clip(scores, 1, 5)  # Ensure within 1-5 range

            ax1.hist(scores, bins=20, edgecolor='black', alpha=0.7)
            ax1.axvline(trust_scores['mean'], color='red', linestyle='--',
                        label=f'Mean: {trust_scores["mean"]:.2f}')
            ax1.set_xlabel('Trust Score')
            ax1.set_ylabel('Count')
            ax1.set_title('Trust Score Distribution')
            ax1.set_xlim(1, 5)
            ax1.legend()

        # Cognitive dimensions radar chart
        if cognitive_alignment:
            dimensions = list(cognitive_alignment.keys())
            values = [cognitive_alignment[d]['mean'] for d in dimensions]

            # Number of variables
            N = len(dimensions)

            # Angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            # Initialize spider plot
            ax2 = plt.subplot(122, projection='polar')

            # Plot data
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2)
            ax2.fill(angles, values, alpha=0.25)

            # Add labels
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(dimensions)
            ax2.set_ylim(0, 5)
            ax2.set_title('Cognitive Alignment Dimensions')
            ax2.grid(True)

        plt.tight_layout()
        self._save_plot('trust_scores.png')

    def plot_performance_radar(self, overall_metrics: Dict[str, Any]):
        """Plot overall performance radar chart."""
        summary = overall_metrics.get('summary', {})
        targets_met = overall_metrics.get('targets_met', {})

        # Prepare data for radar chart
        metrics = ['F1 Score', 'Response Time\nReduction', 'Trust Score', 'Success Rate']
        values = [
            summary.get('classification_f1', 0),
            summary.get('response_time_reduction', 0),
            summary.get('average_trust_score', 0) / 5,  # Normalize to 0-1
            summary.get('success_rate', 0)
        ]

        # Target values (normalized to 0-1)
        targets = [
            targets_met.get('f1_score', {}).get('target', 0.85),
            targets_met.get('response_time_reduction', {}).get('target', 0.6),
            targets_met.get('trust_score', {}).get('target', 4.0) / 5,
            0.9  # Assumed target for success rate
        ]

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]
        targets += targets[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot actual values
        ax.plot(angles, values, 'o-', linewidth=2, label='Actual', color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')

        # Plot target values
        ax.plot(angles, targets, 'o--', linewidth=2, label='Target', color='red')
        ax.fill(angles, targets, alpha=0.1, color='red')

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('A²IR Framework Performance Overview', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.grid(True)

        # Add value labels
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
            ax.text(angle, value + 0.05, f'{value:.2f}',
                    ha='center', va='center', size=10)

        plt.tight_layout()
        self._save_plot('performance_radar.png')

    def _save_plot(self, filename: str):
        """Save plot to file."""
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.debug(f"Plot saved: {filepath}")