# src/neural_components/alert_classification/classifier.py

import logging
import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline # Pipeline might be useful with ColumnTransformer
# from sklearn.compose import ColumnTransformer # For more robust mixed feature handling
# from sklearn.preprocessing import StandardScaler, OneHotEncoder # For numerical/categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# For future use with transformers
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

logger = logging.getLogger(__name__)


class AlertClassifier:
    """
    Neural component for alert classification/triage in the A²IR framework.
    This implements the Nt component in the neurosymbolic integration pathway:
    A²IR = Nt[S(KG, R, I)(Ni)]

    The classifier processes security alerts and classifies them into alert types
    (phishing, malware, DDoS, insider threat) with confidence scores.
    """

    def __init__(self, model_config: Dict[str, Any] = None):
        """
        Initialize the alert classifier.
        Args:
            model_config: Configuration parameters for the classifier
        """
        self.config = model_config or {}
        self.model_type = self.config.get('model_type', 'random_forest')  # Default model type
        # Corrected model_path logic: use 'path' from model_config, then fallback
        self.model_save_dir = self.config.get('path', 'models/alert_classifier')  # Directory to save/load
        self.model = None
        self.classes_ = []  # Changed from self.classes to follow sklearn convention
        self.feature_names_in_ = []  # Changed from self.features
        self.vectorizer = None
        self.last_training_time = None
        self.numerical_feature_columns_ = []  # To store which columns were treated as numeric
        self.text_feature_columns_ = []  # To store which columns were treated as text

        logger.info(f"Initialized AlertClassifier with model type '{self.model_type}'")

    def _extract_features(self, alert_data: Dict[str, Any]) -> pd.Series:
        """
        Extract features from standardized A²IR alert data.
        Now prioritizes 'cic_features' if available.
        Args:
            alert_data: Standardized A²IR alert data dictionary (output from a preprocessor)
                        Expected to have a 'data' key, which might contain 'cic_features'.

        Returns:
            A pandas Series of extracted features for a single alert.
        """
        features = {}
        source_alert_data = alert_data.get('data', {})  # Get the inner 'data' dictionary

        # Prioritize CIC_IDS2017 features if present
        cic_features = source_alert_data.get('cic_features')
        if cic_features and isinstance(cic_features, dict):
            for col, val in cic_features.items():
                # Attempt to convert to numeric, fallback to 0 if not possible or NaN
                # More robust NaN/inf handling should be in CICIDSPreprocessor
                try:
                    if pd.isna(val) or np.isinf(val):
                        features[col] = 0.0  # Or mean/median from training set
                    else:
                        features[col] = float(val)
                except (ValueError, TypeError):
                    features[col] = 0.0  # Fallback for non-numeric/problematic values
            # Retain some generic features if not covered by cic_features or for other datasets
            message = str(source_alert_data.get('message', '')).lower()
            features['message_len'] = len(message)  # Example basic text feature
        else:
            # Fallback to original generic feature extraction if 'cic_features' is not found
            features['timestamp'] = source_alert_data.get('timestamp', datetime.now().timestamp())
            # Simple IP to int might not be robust, consider one-hot encoding or embedding for IPs later
            features['source_ip_numeric'] = int(
                str(source_alert_data.get('source_ip', '0.0.0.0')).replace('.', '')) % 10000
            features['dest_ip_numeric'] = int(str(source_alert_data.get('dest_ip', '0.0.0.0')).replace('.', '')) % 10000
            features['protocol_numeric'] = hash(source_alert_data.get('protocol', '')) % 100

            message = str(source_alert_data.get('message', '')).lower()
            features['message_text'] = message  # Keep the text for TF-IDF if used

            email_data = source_alert_data.get('email', {})
            features['has_attachment'] = 1 if email_data.get('attachments') else 0
            features['sender_domain_hash'] = hash(
                email_data.get('sender', '').split('@')[-1] if '@' in email_data.get('sender', '') else '') % 1000
            features['url_count_email'] = len(email_data.get('urls', []))
            features['email_subject_text'] = str(email_data.get('subject', '')).lower()

            network_data = source_alert_data.get('network', {})
            features['packet_count'] = network_data.get('packet_count', 0)
            features['byte_count'] = network_data.get('byte_count', 0)
            features['duration'] = network_data.get('duration', 0)
            # Example: Convert list of ports to a feature, e.g., count of common ports
            features['port_count'] = len(network_data.get('ports', []))

        return pd.Series(features)

    def _prepare_training_data(self, alerts: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from a list of standardized A²IR alert dictionaries.
        Args:
            alerts: List of alert dictionaries, each expected to have 'data' and 'type' (as label) keys.
        Returns:
            Tuple of features DataFrame (X) and labels Series (y).
        """
        extracted_features_list = []
        labels = []

        for alert in alerts:
            if 'data' not in alert or 'type' not in alert:  # 'type' is the target label
                logger.warning(f"Skipping alert missing 'data' or 'type' (label): {alert.get('id', 'unknown_id')}")
                continue

            # The 'data' field from the standardized alert is passed to _extract_features
            features_series = self._extract_features(alert)  # alert directly IS the standardized alert
            extracted_features_list.append(features_series)
            labels.append(alert['type'])  # 'type' is our ground truth label from preprocessor

        if not extracted_features_list:
            logger.error("No features could be extracted from the provided alerts.")
            return pd.DataFrame(), pd.Series(dtype='object')

        X = pd.DataFrame(extracted_features_list)
        y = pd.Series(labels)

        # Identify numerical and text columns (simple heuristic for this example)
        # In a more robust scenario, this would be predefined or inferred more carefully.
        self.numerical_feature_columns_ = []
        self.text_feature_columns_ = []

        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
                # Check if it's predominantly text or could be categorical
                # For now, assume specific column names are text
                if col in ['message_text', 'email_subject_text']:  # Add other known text fields
                    self.text_feature_columns_.append(col)
                else:  # Treat other objects/strings as potentially categorical or needing specific handling
                    # For simplicity here, we might ignore them or try to hash/encode if this list grows
                    logger.debug(
                        f"Column '{col}' is object type, will be filled with empty string for TFIDF or needs specific handling.")
                    X[col] = X[col].astype(str).fillna('')  # Ensure it's string for TFIDF
                    self.text_feature_columns_.append(col)  # Or decide to handle differently
            else:  # Assume numeric (int, float, bool which becomes 0/1)
                self.numerical_feature_columns_.append(col)

        # Fill NaNs for numerical columns (should ideally be done more intelligently, e.g., imputation)
        X[self.numerical_feature_columns_] = X[self.numerical_feature_columns_].fillna(0)
        # Fill NaNs for text columns with empty string
        X[self.text_feature_columns_] = X[self.text_feature_columns_].fillna('')

        # Store feature names that the model will be trained on
        # This will be a combination after vectorization if text features are used
        # self.feature_names_in_ will be set after fitting vectorizer/preprocessor
        logger.info(f"Prepared data with X shape: {X.shape}, y shape: {y.shape}")
        logger.debug(f"Identified numerical columns: {self.numerical_feature_columns_}")
        logger.debug(f"Identified text columns: {self.text_feature_columns_}")

        return X, y

    def train(self, training_data: List[Dict[str, Any]], validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the alert classifier.
        Args:
            training_data: List of alert dictionaries with 'data' and 'type' (as label) keys.
            validation_split: Proportion of data to use for validation.
        Returns:
            Dictionary containing training metrics.
        """
        logger.info(f"Training alert classifier on {len(training_data)} samples")
        if not training_data:
            logger.error("No training data provided")
            return {"error": "No training data provided"}

        X_df, y_series = self._prepare_training_data(training_data)
        if X_df.empty or y_series.empty:
            logger.error("Failed to prepare training data or data is empty after preparation.")
            return {"error": "Failed to prepare training data or data is empty"}

        self.classes_ = list(y_series.unique())
        logger.info(f"Training for {len(self.classes_)} alert classes: {self.classes_}")

        # Stratified split
        # Ensure there are enough samples for each class for stratification
        min_class_count = y_series.value_counts().min()
        n_splits_for_stratify = min(int(min_class_count), 2)  # At least 2 if possible, else 1 (no split)

        if validation_split > 0 and len(X_df) > 1 and n_splits_for_stratify > 1:
            try:
                X_train_df, X_val_df, y_train, y_val = train_test_split(
                    X_df, y_series, test_size=validation_split, stratify=y_series, random_state=42
                )
            except ValueError as e:  # Happens if a class has only 1 member for stratify
                logger.warning(f"Stratified split failed ({e}), falling back to non-stratified split for validation.")
                X_train_df, X_val_df, y_train, y_val = train_test_split(
                    X_df, y_series, test_size=validation_split, random_state=42
                )
        else:
            logger.info(
                "Validation split is 0 or not enough samples/classes for stratification. Training on full dataset provided.")
            X_train_df, y_train = X_df, y_series
            X_val_df, y_val = pd.DataFrame(), pd.Series(dtype='object')  # Empty validation set

        if self.model_type == 'random_forest':
            X_train_processed = None
            X_val_processed = None

            # Handle text features if any are identified and should be used
            if self.text_feature_columns_:
                logger.info(f"Processing text features: {self.text_feature_columns_}")
                # Combine all identified text columns into a single text series for TF-IDF
                X_train_text_combined_series = X_train_df[self.text_feature_columns_].agg(' '.join, axis=1)

                self.vectorizer = TfidfVectorizer(
                    max_features=self.config.get('tfidf_max_features', 5000),
                    ngram_range=tuple(self.config.get('tfidf_ngram_range', (1, 1)))
                )
                X_train_text_vec = self.vectorizer.fit_transform(X_train_text_combined_series)
                X_train_text_dense = X_train_text_vec.toarray()

                # Store vectorizer feature names
                self.feature_names_in_ = list(self.vectorizer.get_feature_names_out())

                if not X_val_df.empty:
                    X_val_text_combined_series = X_val_df[self.text_feature_columns_].agg(' '.join, axis=1)
                    X_val_text_vec = self.vectorizer.transform(X_val_text_combined_series)
                    X_val_text_dense = X_val_text_vec.toarray()
            else:
                X_train_text_dense = np.array([[]] * len(X_train_df))  # Empty array with correct number of rows
                if not X_val_df.empty:
                    X_val_text_dense = np.array([[]] * len(X_val_df))
                else:
                    X_val_text_dense = np.array([])

            # Handle numerical features
            if self.numerical_feature_columns_:
                logger.info(f"Processing numerical features: {self.numerical_feature_columns_}")
                X_train_numeric = X_train_df[self.numerical_feature_columns_].values
                # Add numerical feature names to self.feature_names_in_ IF text features were not primary
                if not self.text_feature_columns_:  # If only numerical features
                    self.feature_names_in_ = list(self.numerical_feature_columns_)
                elif X_train_text_dense.shape[1] > 0:  # If text features were processed, append numerical names
                    self.feature_names_in_.extend(self.numerical_feature_columns_)
                else:  # Only numerical if text dense is empty
                    self.feature_names_in_ = list(self.numerical_feature_columns_)

                if not X_val_df.empty:
                    X_val_numeric = X_val_df[self.numerical_feature_columns_].values
                else:
                    X_val_numeric = np.array([])
            else:
                X_train_numeric = np.array([[]] * len(X_train_df))
                if not X_val_df.empty:
                    X_val_numeric = np.array([[]] * len(X_val_df))
                else:
                    X_val_numeric = np.array([])
                if not self.text_feature_columns_:  # No features at all
                    logger.error("No text or numerical features found for training.")
                    return {"error": "No usable features found for training."}

            # Combine features: hstack requires both to have same number of rows
            if X_train_text_dense.shape[1] > 0 and X_train_numeric.shape[1] > 0:
                X_train_processed = np.hstack((X_train_text_dense, X_train_numeric))
                if not X_val_df.empty: X_val_processed = np.hstack((X_val_text_dense, X_val_numeric))
            elif X_train_text_dense.shape[1] > 0:  # Only text
                X_train_processed = X_train_text_dense
                if not X_val_df.empty: X_val_processed = X_val_text_dense
            elif X_train_numeric.shape[1] > 0:  # Only numeric
                X_train_processed = X_train_numeric
                if not X_val_df.empty: X_val_processed = X_val_numeric
            else:
                logger.error("No features processed for training (both text and numeric are empty).")
                return {"error": "No features processed."}

            logger.info(f"Training Random Forest classifier with {X_train_processed.shape[1]} processed features.")
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', 20),
                random_state=self.config.get('random_state', 42),
                class_weight=self.config.get('rf_class_weight', 'balanced')  # Good for imbalanced data
            )
            self.model.fit(X_train_processed, y_train)

            metrics = {"train_samples": len(X_train_df)}
            if not X_val_df.empty and X_val_processed is not None and X_val_processed.shape[0] > 0:
                y_pred_val = self.model.predict(X_val_processed)
                f1_val = f1_score(y_val, y_pred_val, average='weighted', zero_division=0)
                metrics["val_samples"] = len(X_val_df)
                metrics["f1_score_val"] = f1_val
                metrics["report_val"] = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)
                logger.info(f"Validation F1 score: {f1_val:.4f}")
            else:
                logger.info("No validation data to evaluate.")
                metrics["f1_score_val"] = None
                metrics["report_val"] = None

            self.last_training_time = datetime.now()
            metrics["classes"] = self.classes_
            logger.info("Completed Random Forest training.")
            return metrics

        # ... (elif self.model_type == 'transformer' placeholder) ...
        else:
            logger.error(f"Unsupported model type: {self.model_type}")
            return {"error": f"Unsupported model type: {self.model_type}"}

    def classify(self, alert_data_standardized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single standardized A²IR security alert.
        Args:
            alert_data_standardized: A single standardized alert dictionary (output from preprocessor).
        Returns:
            Dictionary with classification result and confidence.
        """
        if not self.model:
            logger.error("Model not trained, cannot classify alert.")
            return {"error": "Model not trained", "classification": None, "confidence": 0.0, "probabilities": {}}

        try:
            # Extract features using the same logic as in _prepare_training_data (for a single item)
            # The _extract_features method expects the 'data' sub-dictionary of a standardized alert
            features_series = self._extract_features(alert_data_standardized)

            # Create a DataFrame from the Series for consistency with training feature names
            X_single_df = pd.DataFrame([features_series])

            # Ensure columns are in the same order as during training and fill missing
            # This step is crucial if feature_names_in_ was set based on combined text+numeric
            # For now, we rely on the order from _prepare_training_data and handle text/numeric separately.

            X_single_processed = None

            if self.text_feature_columns_ and self.vectorizer:
                X_single_text_combined_series = X_single_df[self.text_feature_columns_].fillna('').agg(' '.join, axis=1)
                X_single_text_vec = self.vectorizer.transform(X_single_text_combined_series)
                X_single_text_dense = X_single_text_vec.toarray()
            else:  # No text features were trained or no vectorizer
                X_single_text_dense = np.array([[]])  # Empty array, ensure correct shape for hstack later if needed

            if self.numerical_feature_columns_:
                # Ensure all numerical columns seen during training are present, fill with 0 if missing
                for col in self.numerical_feature_columns_:
                    if col not in X_single_df.columns:
                        X_single_df[col] = 0
                X_single_numeric = X_single_df[self.numerical_feature_columns_].fillna(0).values
            else:  # No numerical features were trained
                X_single_numeric = np.array([[]])

            # Combine based on what was trained
            if X_single_text_dense.shape[1] > 0 and X_single_numeric.shape[1] > 0:
                X_single_processed = np.hstack((X_single_text_dense, X_single_numeric))
            elif X_single_text_dense.shape[1] > 0:
                X_single_processed = X_single_text_dense
            elif X_single_numeric.shape[1] > 0:
                X_single_processed = X_single_numeric
            else:
                logger.error("No features could be processed for classification.")
                return {"error": "Feature processing failed", "classification": None, "confidence": 0.0,
                        "probabilities": {}}

            prediction = self.model.predict(X_single_processed)[0]
            probabilities = self.model.predict_proba(X_single_processed)[0]
            confidence = float(np.max(probabilities))  # Ensure it's Python float

            # Make sure self.classes_ is populated (should be after training)
            # And that model.classes_ (from scikit-learn model) matches your self.classes_ order
            prob_dict = {cls_name: float(prob) for cls_name, prob in zip(self.model.classes_, probabilities)}

            return {
                "classification": prediction,
                "confidence": confidence,
                "probabilities": prob_dict
            }

        except Exception as e:
            logger.error(f"Error classifying alert: {e}", exc_info=True)
            return {"error": str(e), "classification": None, "confidence": 0.0, "probabilities": {}}

    def save_model(self, custom_dir_path: str = None) -> bool:
        """
        Save the trained model and associated preprocessors to disk.
        Args:
            custom_dir_path: Optional custom directory path to save the model components.
        Returns:
            True if saved successfully, False otherwise.
        """
        if not self.model:
            logger.error("No trained model to save.")
            return False

        save_directory = custom_dir_path or self.model_save_dir
        os.makedirs(save_directory, exist_ok=True)

        model_file_path = os.path.join(save_directory, 'alert_classifier_model.pkl')
        vectorizer_file_path = os.path.join(save_directory, 'tfidf_vectorizer.pkl')
        metadata_file_path = os.path.join(save_directory, 'classifier_metadata.json')

        try:
            with open(model_file_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved RandomForest model to {model_file_path}")

            if self.vectorizer:
                with open(vectorizer_file_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                logger.info(f"Saved TfidfVectorizer to {vectorizer_file_path}")

            metadata = {
                "classes_": self.classes_,
                "feature_names_in_": self.feature_names_in_,  # Full list of feature names model expects
                "numerical_feature_columns_": self.numerical_feature_columns_,
                "text_feature_columns_": self.text_feature_columns_,
                "model_type": self.model_type,
                "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None
            }
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved classifier metadata to {metadata_file_path}")

            return True
        except Exception as e:
            logger.error(f"Error saving model components: {e}", exc_info=True)
            return False

    def load_model(self, custom_dir_path: str = None) -> bool:
        """
        Load a trained model and associated preprocessors from disk.
        Args:
            custom_dir_path: Optional custom directory path to load the model from.
        Returns:
            True if loaded successfully, False otherwise.
        """
        load_directory = custom_dir_path or self.model_save_dir

        model_file_path = os.path.join(load_directory, 'alert_classifier_model.pkl')
        vectorizer_file_path = os.path.join(load_directory, 'tfidf_vectorizer.pkl')
        metadata_file_path = os.path.join(load_directory, 'classifier_metadata.json')

        if not os.path.exists(model_file_path) or not os.path.exists(metadata_file_path):
            logger.error(f"Model or metadata file not found in directory: {load_directory}")
            return False

        try:
            with open(model_file_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded RandomForest model from {model_file_path}")

            if os.path.exists(vectorizer_file_path):
                with open(vectorizer_file_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info(f"Loaded TfidfVectorizer from {vectorizer_file_path}")
            else:
                self.vectorizer = None  # Ensure it's None if file doesn't exist
                logger.info("No TfidfVectorizer file found to load.")

            with open(metadata_file_path, 'r') as f:
                metadata = json.load(f)

            self.classes_ = metadata.get("classes_", [])
            self.feature_names_in_ = metadata.get("feature_names_in_", [])
            self.numerical_feature_columns_ = metadata.get("numerical_feature_columns_", [])
            self.text_feature_columns_ = metadata.get("text_feature_columns_", [])
            self.model_type = metadata.get("model_type", "random_forest")
            time_str = metadata.get("last_training_time")
            self.last_training_time = datetime.fromisoformat(time_str) if time_str else None

            logger.info(f"Loaded classifier metadata. Model type: {self.model_type}, Classes: {len(self.classes_)}")
            return True

        except Exception as e:
            logger.error(f"Error loading model components: {e}", exc_info=True)
            return False