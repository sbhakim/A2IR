# src/trust_framework/trust_score.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TrustScore:
    """
    Implements the Trust Score framework for A²IR.

    The Trust Score (TS) combines algorithmic fidelity and human-perceived 
    trustworthiness using the formula:
    TS(e, c) = α * HR(e) + (1 - α) * SF(e, c)

    Where:
    - HR(e) is the Human Rating of explanation quality
    - SF(e, c) is the SHAP Fidelity measure
    - α is the weighting parameter (default 0.6)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Trust Score calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.alpha = self.config.get('alpha_weighting', 0.6)

        # Storage for ratings history
        self.human_ratings_history = []
        self.fidelity_scores_history = []
        self.trust_scores_history = []

        # Cognitive alignment metrics
        self.cognitive_dimensions = [
            'clarity',
            'completeness',
            'actionability',
            'coherence',
            'relevance'
        ]

        logger.info(f"Trust Score initialized with alpha={self.alpha}")

    def calculate_trust_score(self,
                              human_rating: float,
                              system_fidelity: float,
                              alpha: Optional[float] = None) -> float:
        """
        Calculate the Trust Score.

        Args:
            human_rating: Human rating (1-5 scale)
            system_fidelity: System fidelity score (0-1 scale)
            alpha: Optional override for alpha weighting

        Returns:
            Trust Score (1-5 scale)
        """
        if alpha is None:
            alpha = self.alpha

        # Validate inputs
        human_rating = max(1.0, min(5.0, human_rating))
        system_fidelity = max(0.0, min(1.0, system_fidelity))

        # Convert system fidelity to 1-5 scale
        system_fidelity_scaled = 1 + (system_fidelity * 4)

        # Calculate Trust Score
        trust_score = alpha * human_rating + (1 - alpha) * system_fidelity_scaled

        # Store in history
        self.human_ratings_history.append(human_rating)
        self.fidelity_scores_history.append(system_fidelity)
        self.trust_scores_history.append(trust_score)

        return trust_score

    def calculate_shap_fidelity(self,
                                explanation: Dict[str, Any],
                                classification: Dict[str, Any]) -> float:
        """
        Calculate SHAP fidelity score for an explanation.

        This measures how well the explanation aligns with the model's decision.

        Args:
            explanation: The explanation provided
            classification: The classification result

        Returns:
            Fidelity score (0-1)
        """
        # Extract feature importances from explanation
        feature_importance = self._extract_feature_importance(explanation)

        # Extract confidence scores from classification
        classification_confidence = classification.get('confidence', 0)
        predicted_class = classification.get('classification')

        # Calculate alignment between important features and classification
        if not feature_importance:
            return classification_confidence  # Fallback to classification confidence

        # Check if highly important features align with the predicted class
        alignment_score = 0.0
        total_weight = 0.0

        for feature, importance in feature_importance.items():
            # Check if this feature supports the classification
            if self._feature_supports_classification(feature, predicted_class, explanation):
                alignment_score += importance
            total_weight += importance

        if total_weight > 0:
            fidelity = alignment_score / total_weight
        else:
            fidelity = classification_confidence

        # Weight by classification confidence
        fidelity = fidelity * classification_confidence

        return min(1.0, max(0.0, fidelity))

    def _extract_feature_importance(self, explanation: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract feature importance scores from explanation.

        Args:
            explanation: The explanation provided

        Returns:
            Dictionary of feature names to importance scores
        """
        feature_importance = {}

        # Look for feature importance in different places
        if 'feature_importance' in explanation:
            feature_importance = explanation['feature_importance']
        elif 'reasoning_path' in explanation:
            # Extract from reasoning path
            for step in explanation['reasoning_path']:
                if 'features' in step:
                    for feature, value in step['features'].items():
                        feature_importance[feature] = feature_importance.get(feature, 0) + 0.1
        elif 'integration_trace' in explanation:
            # Extract from integration trace
            for trace in explanation['integration_trace']:
                if 'data' in trace and isinstance(trace['data'], dict):
                    if 'features' in trace['data']:
                        for feature in trace['data']['features']:
                            feature_importance[feature] = feature_importance.get(feature, 0) + 0.1

        # Normalize importance scores
        if feature_importance:
            total = sum(feature_importance.values())
            if total > 0:
                return {k: v / total for k, v in feature_importance.items()}

        return feature_importance

    def _feature_supports_classification(self,
                                         feature: str,
                                         predicted_class: str,
                                         explanation: Dict[str, Any]) -> bool:
        """
        Determine if a feature supports the given classification.

        Args:
            feature: Feature name
            predicted_class: The predicted class
            explanation: The explanation

        Returns:
            True if feature supports classification, False otherwise
        """
        # Simple heuristic mappings
        feature_class_mappings = {
            'phishing': ['suspicious_sender', 'malicious_url', 'spoofed_domain', 'urgent_language'],
            'malware': ['suspicious_process', 'registry_modification', 'encrypted_traffic'],
            'ddos': ['traffic_spike', 'bandwidth_saturation', 'connection_surge'],
            'insider_threat': ['unusual_access_time', 'bulk_data_access', 'data_exfiltration']
        }

        # Check if feature is in the expected features for this class
        expected_features = feature_class_mappings.get(predicted_class, [])

        for expected in expected_features:
            if expected in feature.lower():
                return True

        # Check in reasoning path
        if 'reasoning_path' in explanation:
            for step in explanation['reasoning_path']:
                if feature in str(step) and predicted_class in str(step):
                    return True

        return False

    def collect_human_rating(self,
                             explanation: Dict[str, Any],
                             rating_dimensions: Optional[Dict[str, float]] = None) -> float:
        """
        Collect human rating for an explanation.

        In a real system, this would interface with a UI for analysts to rate.
        For now, this simulates the rating process.

        Args:
            explanation: The explanation to rate
            rating_dimensions: Optional specific dimension ratings

        Returns:
            Overall human rating (1-5)
        """
        if rating_dimensions:
            # Calculate average of dimension ratings
            ratings = []
            for dimension in self.cognitive_dimensions:
                if dimension in rating_dimensions:
                    ratings.append(rating_dimensions[dimension])

            if ratings:
                return sum(ratings) / len(ratings)

        # Simulate rating based on explanation quality heuristics
        base_rating = 3.0  # Neutral baseline

        # Adjust based on explanation characteristics
        if 'reasoning_path' in explanation and len(explanation['reasoning_path']) > 0:
            base_rating += 0.5

        if 'identified_threats' in explanation and explanation['identified_threats']:
            base_rating += 0.5

        if 'recommended_actions' in explanation and explanation['recommended_actions']:
            base_rating += 0.5

        if 'integration_trace' in explanation and len(explanation['integration_trace']) > 3:
            base_rating += 0.3

        # Ensure within bounds
        return max(1.0, min(5.0, base_rating))

    def evaluate_cognitive_alignment(self,
                                     explanation: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate explanation against cognitive dimensions.

        Args:
            explanation: The explanation to evaluate

        Returns:
            Scores for each cognitive dimension
        """
        scores = {}

        # Clarity - Is the explanation easy to understand?
        clarity_score = 3.0
        if 'reasoning_path' in explanation:
            # Check for clear step-by-step reasoning
            steps = explanation['reasoning_path']
            if all('description' in step for step in steps):
                clarity_score += 1.0
            if len(steps) > 0 and len(steps) < 10:  # Not too many steps
                clarity_score += 0.5
        scores['clarity'] = min(5.0, clarity_score)

        # Completeness - Does it cover all important aspects?
        completeness_score = 2.0
        required_elements = ['classification', 'confidence', 'reasoning_path', 'recommended_actions']
        present_elements = sum(1 for elem in required_elements if elem in explanation)
        completeness_score += (present_elements / len(required_elements)) * 2
        scores['completeness'] = min(5.0, completeness_score)

        # Actionability - Does it provide clear next steps?
        actionability_score = 2.0
        if 'recommended_actions' in explanation:
            actions = explanation['recommended_actions']
            if actions:
                actionability_score += 2.0
                if all('priority' in action for action in actions):
                    actionability_score += 1.0
        scores['actionability'] = min(5.0, actionability_score)

        # Coherence - Is the explanation internally consistent?
        coherence_score = 3.0
        if 'integration_trace' in explanation:
            # Check if trace follows logical progression
            trace = explanation['integration_trace']
            expected_order = ['Neural Triage', 'Knowledge Graph', 'Symbolic Reasoning']
            actual_order = [step['component'] for step in trace[:3]]
            if all(exp in actual_order for exp in expected_order):
                coherence_score += 1.0
        scores['coherence'] = min(5.0, coherence_score)

        # Relevance - Is the explanation relevant to the alert?
        relevance_score = 3.0
        if 'alert_type' in explanation and 'classification' in explanation:
            if 'type' in explanation['classification']:
                if explanation['alert_type'] == explanation['classification']['type']:
                    relevance_score += 1.5
        scores['relevance'] = min(5.0, relevance_score)

        return scores

    def calculate_comprehensive_trust_score(self,
                                            explanation: Dict[str, Any],
                                            classification: Dict[str, Any],
                                            human_ratings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive trust score with all dimensions.

        Args:
            explanation: The explanation to evaluate
            classification: The classification result
            human_ratings: Optional human ratings for dimensions

        Returns:
            Comprehensive trust evaluation
        """
        # Calculate SHAP fidelity
        shap_fidelity = self.calculate_shap_fidelity(explanation, classification)

        # Get cognitive dimension scores
        cognitive_scores = self.evaluate_cognitive_alignment(explanation)

        # Get human rating
        if human_ratings:
            human_rating = self.collect_human_rating(explanation, human_ratings)
        else:
            # Use cognitive scores as proxy
            human_rating = sum(cognitive_scores.values()) / len(cognitive_scores)

        # Calculate final trust score
        trust_score = self.calculate_trust_score(human_rating, shap_fidelity)

        # Compile comprehensive result
        result = {
            'trust_score': trust_score,
            'human_rating': human_rating,
            'shap_fidelity': shap_fidelity,
            'cognitive_scores': cognitive_scores,
            'timestamp': datetime.now().isoformat(),
            'explanation_id': explanation.get('incident_id', 'unknown')
        }

        return result

    def analyze_trust_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in trust scores over time.

        Returns:
            Analysis of trust score trends
        """
        if not self.trust_scores_history:
            return {"error": "No trust scores recorded yet"}

        trust_scores = np.array(self.trust_scores_history)
        human_ratings = np.array(self.human_ratings_history)
        fidelity_scores = np.array(self.fidelity_scores_history)

        analysis = {
            'num_evaluations': len(trust_scores),
            'trust_score_stats': {
                'mean': float(np.mean(trust_scores)),
                'std': float(np.std(trust_scores)),
                'min': float(np.min(trust_scores)),
                'max': float(np.max(trust_scores))
            },
            'human_rating_stats': {
                'mean': float(np.mean(human_ratings)),
                'std': float(np.std(human_ratings))
            },
            'fidelity_score_stats': {
                'mean': float(np.mean(fidelity_scores)),
                'std': float(np.std(fidelity_scores))
            }
        }

        # Check if trust is improving over time
        if len(trust_scores) > 10:
            recent_scores = trust_scores[-10:]
            older_scores = trust_scores[:-10]
            analysis['trust_trend'] = {
                'recent_mean': float(np.mean(recent_scores)),
                'older_mean': float(np.mean(older_scores)),
                'improving': bool(np.mean(recent_scores) > np.mean(older_scores))
            }

        return analysis

    def export_trust_data(self, filepath: str):
        """
        Export trust score data for analysis.

        Args:
            filepath: Path to save the data
        """
        data = {
            'trust_scores': self.trust_scores_history,
            'human_ratings': self.human_ratings_history,
            'fidelity_scores': self.fidelity_scores_history,
            'alpha_weight': self.alpha,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Trust data exported to {filepath}")


def validate_trust_score_properties():
    """
    Validate theoretical properties of the Trust Score.

    This demonstrates the monotonicity, stability, and sensitivity properties
    claimed in the dissertation.
    """
    trust_calculator = TrustScore()

    logger.info("Validating Trust Score properties...")

    # Test monotonicity
    logger.info("Testing monotonicity...")
    hr_values = [1, 2, 3, 4, 5]
    sf_fixed = 0.8
    scores = []

    for hr in hr_values:
        score = trust_calculator.calculate_trust_score(hr, sf_fixed)
        scores.append(score)

    monotonic_hr = all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))
    logger.info(f"Monotonicity with respect to HR: {monotonic_hr}")

    # Test stability
    logger.info("Testing stability...")
    hr_fixed = 4.0
    sf_values = [0.7, 0.71, 0.72, 0.73, 0.74]
    scores = []

    for sf in sf_values:
        score = trust_calculator.calculate_trust_score(hr_fixed, sf)
        scores.append(score)

    max_diff = max(abs(scores[i] - scores[i + 1]) for i in range(len(scores) - 1))
    stable = max_diff < 0.1  # Small changes in input lead to small changes in output
    logger.info(f"Stability property (max diff = {max_diff}): {stable}")

    # Test sensitivity
    logger.info("Testing sensitivity...")
    hr1 = 5.0
    hr2 = 1.0
    sf = 0.5

    score1 = trust_calculator.calculate_trust_score(hr1, sf)
    score2 = trust_calculator.calculate_trust_score(hr2, sf)

    sensitivity = abs(score1 - score2)
    logger.info(f"Sensitivity to human rating changes: {sensitivity}")

    return {
        'monotonicity': monotonic_hr,
        'stability': stable,
        'sensitivity': sensitivity
    }