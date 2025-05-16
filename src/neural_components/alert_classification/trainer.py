# src/neural_components/alert_classification/trainer.py

import logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from .classifier import AlertClassifier

logger = logging.getLogger(__name__)


class AlertClassifierTrainer:
    """
    Trainer for the AlertClassifier in the A²IR framework.

    Handles data preparation, training, evaluation, and hyperparameter tuning
    for the neural triage component (Nt).
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})

        # Training parameters
        self.batch_size = self.training_config.get('batch_size', 32)
        self.epochs = self.training_config.get('epochs', 10)
        self.validation_split = self.training_config.get('validation_split', 0.2)
        self.cross_validation_folds = self.training_config.get('cv_folds', 5)

        # Output paths
        self.output_dir = self.config.get('output_dir', 'models/alert_classifier')
        self.metrics_dir = os.path.join(self.output_dir, 'metrics')
        self.plots_dir = os.path.join(self.output_dir, 'plots')

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Label encoder for alert types
        self.label_encoder = LabelEncoder()

        logger.info("AlertClassifierTrainer initialized")

    def prepare_training_data(self, data_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Prepare training data from a file or directory.

        Args:
            data_path: Path to training data

        Returns:
            Tuple of (training data, data statistics)
        """
        logger.info(f"Preparing training data from: {data_path}")

        training_data = []

        if os.path.isfile(data_path):
            # Load from single file
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    training_data = json.load(f)
            elif data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                training_data = df.to_dict('records')

        elif os.path.isdir(data_path):
            # Load from directory
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            training_data.extend(file_data)
                        else:
                            training_data.append(file_data)

        else:
            raise ValueError(f"Invalid data path: {data_path}")

        # Validate data format
        valid_data = []
        for item in training_data:
            if self._validate_training_item(item):
                valid_data.append(item)

        # Generate statistics
        stats = self._generate_data_statistics(valid_data)

        logger.info(f"Prepared {len(valid_data)} training samples")
        logger.info(f"Data statistics: {stats}")

        return valid_data, stats

    def _validate_training_item(self, item: Dict[str, Any]) -> bool:
        """Validate a training data item."""
        # Check required fields
        required_fields = ['data', 'label']

        for field in required_fields:
            if field not in item:
                return False

        # Check data structure
        if not isinstance(item['data'], dict):
            return False

        # Check label
        if not item['label']:
            return False

        return True

    def _generate_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the training data."""
        stats = {
            'total_samples': len(data),
            'label_distribution': {},
            'feature_statistics': {}
        }

        # Label distribution
        labels = [item['label'] for item in data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        stats['label_distribution'] = dict(zip(unique_labels, counts.tolist()))

        # Feature statistics
        all_features = []
        for item in data:
            if 'data' in item:
                all_features.extend(item['data'].keys())

        unique_features, feature_counts = np.unique(all_features, return_counts=True)
        stats['feature_statistics'] = {
            'unique_features': len(unique_features),
            'most_common_features': dict(zip(
                unique_features[np.argsort(feature_counts)[-10:]],
                feature_counts[np.argsort(feature_counts)[-10:]].tolist()
            ))
        }

        return stats

    def train_model(self,
                    training_data: List[Dict[str, Any]],
                    validation_data: Optional[List[Dict[str, Any]]] = None) -> Tuple[AlertClassifier, Dict[str, Any]]:
        """
        Train the alert classifier model.

        Args:
            training_data: List of training samples
            validation_data: Optional list of validation samples

        Returns:
            Tuple of (trained classifier, training metrics)
        """
        logger.info("Starting model training...")

        # Initialize classifier
        classifier = AlertClassifier(self.model_config)

        # If no validation data provided, split from training data
        if validation_data is None:
            training_data, validation_data = train_test_split(
                training_data,
                test_size=self.validation_split,
                stratify=[item['label'] for item in training_data],
                random_state=42
            )

        # Train the model
        training_result = classifier.train(training_data, validation_split=0.0)

        # Evaluate on validation data
        validation_metrics = self.evaluate_model(classifier, validation_data)

        # Combine metrics
        metrics = {
            'training': training_result,
            'validation': validation_metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save metrics
        metrics_file = os.path.join(
            self.metrics_dir,
            f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Metrics saved to {metrics_file}")

        return classifier, metrics

    def evaluate_model(self,
                       classifier: AlertClassifier,
                       test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            classifier: Trained AlertClassifier
            test_data: List of test samples

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model on {len(test_data)} samples...")

        predictions = []
        true_labels = []
        confidence_scores = []

        for item in test_data:
            result = classifier.classify(item['data'])
            predictions.append(result['classification'])
            true_labels.append(item['label'])
            confidence_scores.append(result['confidence'])

        # Calculate metrics
        metrics = {
            'accuracy': sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions),
            'classification_report': classification_report(
                true_labels, predictions, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            }
        }

        # Generate plots
        self._plot_confusion_matrix(true_labels, predictions)
        self._plot_confidence_distribution(confidence_scores, predictions, true_labels)

        return metrics

    def cross_validate(self,
                       data: List[Dict[str, Any]],
                       n_splits: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            data: Training data
            n_splits: Number of CV folds (defaults to config value)

        Returns:
            Cross-validation results
        """
        if n_splits is None:
            n_splits = self.cross_validation_folds

        logger.info(f"Starting {n_splits}-fold cross-validation...")

        # Prepare data
        X = [item['data'] for item in data]
        y = [item['label'] for item in data]

        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_splits}")

            # Split data
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            # Train model
            classifier = AlertClassifier(self.model_config)
            training_result = classifier.train(train_data, validation_split=0.0)

            # Evaluate
            val_metrics = self.evaluate_model(classifier, val_data)

            cv_results.append({
                'fold': fold + 1,
                'training': training_result,
                'validation': val_metrics
            })

        # Aggregate results
        aggregated_metrics = self._aggregate_cv_results(cv_results)

        # Save CV results
        cv_file = os.path.join(
            self.metrics_dir,
            f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(cv_file, 'w') as f:
            json.dump({
                'fold_results': cv_results,
                'aggregated': aggregated_metrics
            }, f, indent=2)

        logger.info(f"Cross-validation complete. Results saved to {cv_file}")

        return aggregated_metrics

    def _aggregate_cv_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        aggregated = {
            'accuracy': {
                'mean': np.mean([r['validation']['accuracy'] for r in cv_results]),
                'std': np.std([r['validation']['accuracy'] for r in cv_results])
            },
            'f1_scores': {}
        }

        # Aggregate per-class metrics
        all_classes = set()
        for result in cv_results:
            all_classes.update(result['validation']['classification_report'].keys())

        for class_name in all_classes:
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue

            f1_scores = []
            for result in cv_results:
                if class_name in result['validation']['classification_report']:
                    f1_scores.append(
                        result['validation']['classification_report'][class_name]['f1-score']
                    )

            if f1_scores:
                aggregated['f1_scores'][class_name] = {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores)
                }

        return aggregated

    def hyperparameter_search(self,
                              data: List[Dict[str, Any]],
                              param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Perform hyperparameter search.

        Args:
            data: Training data
            param_grid: Dictionary of parameters to search

        Returns:
            Best parameters and results
        """
        logger.info("Starting hyperparameter search...")

        best_score = 0
        best_params = {}
        results = []

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameters {i + 1}/{len(param_combinations)}: {params}")

            # Update model config with current parameters
            model_config = dict(self.model_config)
            model_config.update(params)

            # Train and evaluate
            classifier = AlertClassifier(model_config)
            cv_scores = []

            # Simple CV for each parameter set
            for train_data, val_data in self._split_data_cv(data, n_splits=3):
                classifier.train(train_data, validation_split=0.0)
                val_metrics = self.evaluate_model(classifier, val_data)
                cv_scores.append(val_metrics['accuracy'])

            avg_score = np.mean(cv_scores)

            results.append({
                'params': params,
                'score': avg_score,
                'cv_scores': cv_scores
            })

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

        # Save results
        search_results = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'timestamp': datetime.now().isoformat()
        }

        results_file = os.path.join(
            self.metrics_dir,
            f"hyperparam_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(results_file, 'w') as f:
            json.dump(search_results, f, indent=2)

        logger.info(f"Hyperparameter search complete. Best params: {best_params}")

        return search_results

    def _generate_param_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools

        keys = param_grid.keys()
        values = param_grid.values()

        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))

        return combinations

    def _split_data_cv(self, data: List[Dict[str, Any]], n_splits: int = 3):
        """Split data for cross-validation."""
        labels = [item['label'] for item in data]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(data, labels):
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
            yield train_data, val_data

    def _plot_confusion_matrix(self, true_labels: List[str], predictions: List[str]):
        """Plot confusion matrix."""
        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        plot_file = os.path.join(
            self.plots_dir,
            f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_file)
        plt.close()

        logger.info(f"Confusion matrix saved to {plot_file}")

    def _plot_confidence_distribution(self,
                                      confidence_scores: List[float],
                                      predictions: List[str],
                                      true_labels: List[str]):
        """Plot confidence distribution."""
        correct_predictions = [c for c, p, t in zip(confidence_scores, predictions, true_labels) if p == t]
        incorrect_predictions = [c for c, p, t in zip(confidence_scores, predictions, true_labels) if p != t]

        plt.figure(figsize=(10, 6))
        plt.hist(correct_predictions, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_predictions, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Distribution by Prediction Correctness')
        plt.legend()

        plot_file = os.path.join(
            self.plots_dir,
            f"confidence_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_file)
        plt.close()

        logger.info(f"Confidence distribution plot saved to {plot_file}")

    def train_final_model(self, data_path: str) -> AlertClassifier:
        """
        Train the final production model.

        Args:
            data_path: Path to training data

        Returns:
            Trained AlertClassifier
        """
        logger.info("Training final production model...")

        # Prepare data
        training_data, stats = self.prepare_training_data(data_path)

        # Optionally perform hyperparameter search
        if self.training_config.get('do_hyperparam_search', False):
            param_grid = self.training_config.get('param_grid', {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None]
            })
            search_results = self.hyperparameter_search(training_data, param_grid)

            # Update config with best parameters
            self.model_config.update(search_results['best_params'])

        # Train final model on all data
        classifier = AlertClassifier(self.model_config)
        training_result = classifier.train(training_data, validation_split=0.0)

        # Save the model
        classifier.save_model()

        # Save training report
        report = {
            'data_statistics': stats,
            'model_config': self.model_config,
            'training_result': training_result,
            'timestamp': datetime.now().isoformat()
        }

        report_file = os.path.join(
            self.output_dir,
            f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Final model trained and saved. Report: {report_file}")

        return classifier


def train(config: Dict[str, Any], data_path: str) -> AlertClassifier:
    """
    Convenience function to train a model.

    Args:
        config: Training configuration
        data_path: Path to training data

    Returns:
        Trained AlertClassifier
    """
    trainer = AlertClassifierTrainer(config)
    return trainer.train_final_model(data_path)