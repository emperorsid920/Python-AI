import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModel, BertTokenizer, BertForSequenceClassification,
    pipeline, Trainer, TrainingArguments
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentModerationModel:
    """
    Main class that orchestrates multiple models for content moderation.
    Handles sentiment analysis, spam detection, and toxicity detection.
    """

    def __init__(self, model_dir: str = "models/"):
        """
        Initialize the content moderation model.

        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize model components
        self.sentiment_model = None
        self.spam_model = None
        self.toxicity_model = None

        # Model configurations
        self.model_configs = {
            'sentiment': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'labels': ['negative', 'neutral', 'positive']
            },
            'spam': {
                'model_name': 'unitary/toxic-bert',
                'threshold': 0.5
            },
            'toxicity': {
                'model_name': 'unitary/toxic-bert',
                'threshold': 0.7
            }
        }

        logger.info(f"ContentModerationModel initialized. Using device: {self.device}")

    def load_pretrained_models(self):
        """
        Load pre-trained models for immediate use.
        This is faster than training from scratch.
        """
        try:
            # Load sentiment analysis model
            logger.info("Loading sentiment analysis model...")
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=self.model_configs['sentiment']['model_name'],
                device=0 if self.device.type == 'cuda' else -1
            )

            # Load toxicity detection model
            logger.info("Loading toxicity detection model...")
            self.toxicity_model = pipeline(
                "text-classification",
                model=self.model_configs['toxicity']['model_name'],
                device=0 if self.device.type == 'cuda' else -1
            )

            # Initialize spam detection (we'll use a simpler approach initially)
            self.spam_model = SpamDetector()

            logger.info("All pre-trained models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading pre-trained models: {e}")
            return False

    def predict_sentiment(self, text: str) -> Dict:
        """
        Predict sentiment for a given text.

        Args:
            text (str): Text to analyze

        Returns:
            Dict: Sentiment prediction with score and label
        """
        try:
            if self.sentiment_model is None:
                raise ValueError("Sentiment model not loaded")

            # Get prediction
            result = self.sentiment_model(text)[0]

            # Normalize the output
            sentiment_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral',
                'POSITIVE': 'positive'
            }

            label = sentiment_mapping.get(result['label'], result['label'].lower())
            score = result['score']

            return {
                'sentiment_label': label,
                'sentiment_score': score,
                'confidence': score
            }

        except Exception as e:
            logger.error(f"Error in sentiment prediction: {e}")
            return {
                'sentiment_label': 'neutral',
                'sentiment_score': 0.5,
                'confidence': 0.0
            }

    def predict_spam(self, text: str, features: Dict = None) -> Dict:
        """
        Predict if text is spam.

        Args:
            text (str): Text to analyze
            features (Dict): Additional features from data processing

        Returns:
            Dict: Spam prediction with probability and classification
        """
        try:
            if self.spam_model is None:
                raise ValueError("Spam model not loaded")

            # Use the spam detector
            result = self.spam_model.predict(text, features)

            return {
                'spam_probability': result['probability'],
                'is_spam': result['is_spam'],
                'spam_reasons': result['reasons']
            }

        except Exception as e:
            logger.error(f"Error in spam prediction: {e}")
            return {
                'spam_probability': 0.0,
                'is_spam': False,
                'spam_reasons': []
            }

    def predict_toxicity(self, text: str) -> Dict:
        """
        Predict toxicity level of text.

        Args:
            text (str): Text to analyze

        Returns:
            Dict: Toxicity prediction with score and classification
        """
        try:
            if self.toxicity_model is None:
                raise ValueError("Toxicity model not loaded")

            # Get prediction
            result = self.toxicity_model(text)[0]

            # The toxic-bert model returns TOXIC or NOT_TOXIC
            is_toxic = result['label'] == 'TOXIC'
            toxicity_score = result['score'] if is_toxic else 1 - result['score']

            return {
                'toxicity_score': toxicity_score,
                'is_toxic': toxicity_score > self.model_configs['toxicity']['threshold'],
                'confidence': result['score']
            }

        except Exception as e:
            logger.error(f"Error in toxicity prediction: {e}")
            return {
                'toxicity_score': 0.0,
                'is_toxic': False,
                'confidence': 0.0
            }

    def predict_all(self, text: str, features: Dict = None) -> Dict:
        """
        Run all predictions on a text.

        Args:
            text (str): Text to analyze
            features (Dict): Additional features from data processing

        Returns:
            Dict: Combined predictions from all models
        """
        # Get all predictions
        sentiment_result = self.predict_sentiment(text)
        spam_result = self.predict_spam(text, features)
        toxicity_result = self.predict_toxicity(text)

        # Combine results
        combined_result = {
            **sentiment_result,
            **spam_result,
            **toxicity_result,
            'model_version': 'v1.0',
            'prediction_timestamp': datetime.now().isoformat()
        }

        # Calculate overall confidence
        confidences = [
            sentiment_result['confidence'],
            spam_result['spam_probability'],
            toxicity_result['confidence']
        ]
        combined_result['confidence_score'] = np.mean(confidences)

        return combined_result

    def batch_predict(self, texts: List[str], features_list: List[Dict] = None) -> List[Dict]:
        """
        Predict on multiple texts efficiently.

        Args:
            texts (List[str]): List of texts to analyze
            features_list (List[Dict]): List of feature dictionaries

        Returns:
            List[Dict]: List of prediction results
        """
        if features_list is None:
            features_list = [None] * len(texts)

        results = []
        for i, (text, features) in enumerate(zip(texts, features_list)):
            try:
                result = self.predict_all(text, features)
                results.append(result)

                # Log progress for large batches
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")

            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                # Add default result for failed predictions
                results.append({
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.5,
                    'spam_probability': 0.0,
                    'is_spam': False,
                    'toxicity_score': 0.0,
                    'is_toxic': False,
                    'confidence_score': 0.0,
                    'error': str(e)
                })

        return results

    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data.

        Args:
            test_data (pd.DataFrame): Test dataset with ground truth labels

        Returns:
            Dict: Evaluation metrics
        """
        predictions = []
        ground_truth = []

        for _, row in test_data.iterrows():
            try:
                # Get prediction
                pred = self.predict_sentiment(row['review_text'])
                predictions.append(pred['sentiment_label'])
                ground_truth.append(row['sentiment_label'])

            except Exception as e:
                logger.error(f"Error evaluating row: {e}")
                predictions.append('neutral')
                ground_truth.append(row['sentiment_label'])

        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )

        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(ground_truth, predictions)
        }

        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.3f}")
        return evaluation_results

    def save_model_info(self, model_info: Dict):
        """
        Save model information and metadata.

        Args:
            model_info (Dict): Model metadata to save
        """
        info_path = os.path.join(self.model_dir, "model_info.json")

        model_info.update({
            'saved_at': datetime.now().isoformat(),
            'device': str(self.device),
            'model_configs': self.model_configs
        })

        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Model info saved to {info_path}")


class SpamDetector:
    """
    Rule-based spam detector that uses heuristics and features.
    This is simpler than training a full ML model but still effective.
    """

    def __init__(self):
        """Initialize the spam detector with rules and thresholds."""
        self.spam_keywords = [
            'free', 'win', 'winner', 'urgent', 'limited time', 'act now',
            'click here', 'earn money', 'make money', 'work from home',
            'guarantee', 'risk free', 'no risk', 'call now', 'buy now',
            'special offer', 'discount', 'save money', 'cash', 'credit',
            'loan', 'debt', 'investment', 'profit', 'income'
        ]

        self.suspicious_patterns = [
            r'\b(?:https?://|www\.)',  # URLs
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'  # Phone numbers
        ]

        logger.info("SpamDetector initialized")

    def predict(self, text: str, features: Dict = None) -> Dict:
        """
        Predict if text is spam based on rules and features.

        Args:
            text (str): Text to analyze
            features (Dict): Additional features from data processing

        Returns:
            Dict: Spam prediction result
        """
        spam_score = 0.0
        reasons = []

        text_lower = text.lower()

        # Check for spam keywords
        keyword_count = sum(1 for keyword in self.spam_keywords if keyword in text_lower)
        if keyword_count > 0:
            keyword_score = min(keyword_count * 0.2, 0.6)
            spam_score += keyword_score
            reasons.append(f"Contains {keyword_count} spam keywords")

        # Check for suspicious patterns
        import re
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text):
                spam_score += 0.3
                reasons.append("Contains suspicious patterns (URLs/emails/phones)")
                break

        # Use features if available
        if features:
            # High caps ratio
            if features.get('caps_ratio', 0) > 0.3:
                spam_score += 0.2
                reasons.append("High capitalization ratio")

            # Many exclamation marks
            if features.get('exclamation_count', 0) > 3:
                spam_score += 0.15
                reasons.append("Excessive exclamation marks")

            # Very short or very long text
            text_length = features.get('text_length', len(text))
            if text_length < 20:
                spam_score += 0.1
                reasons.append("Very short text")
            elif text_length > 1000:
                spam_score += 0.1
                reasons.append("Very long text")

            # High spam keyword count from features
            if features.get('spam_keyword_count', 0) > 2:
                spam_score += 0.2
                reasons.append("High spam keyword count")

        # Normalize score to 0-1 range
        spam_probability = min(spam_score, 1.0)
        is_spam = spam_probability > 0.5

        return {
            'probability': spam_probability,
            'is_spam': is_spam,
            'reasons': reasons
        }


class ModelTrainer:
    """
    Class for training custom models on your specific data.
    This is for advanced users who want to fine-tune models.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the model trainer.

        Args:
            model_name (str): Name of the base model to fine-tune
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        logger.info(f"ModelTrainer initialized with {model_name}")

    def prepare_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple:
        """
        Prepare dataset for training.

        Args:
            train_df (pd.DataFrame): Training data
            val_df (pd.DataFrame): Validation data

        Returns:
            Tuple: Prepared training and validation datasets
        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize texts
        def tokenize_function(examples):
            return self.tokenizer(
                examples['review_text'],
                truncation=True,
                padding=True,
                max_length=512
            )

        # Prepare training data
        train_encodings = tokenize_function(train_df)
        val_encodings = tokenize_function(val_df)

        # Create datasets
        train_dataset = CustomDataset(train_encodings, train_df['sentiment_encoded'].tolist())
        val_dataset = CustomDataset(val_encodings, val_df['sentiment_encoded'].tolist())

        return train_dataset, val_dataset

    def train_sentiment_model(self, train_dataset, val_dataset, num_labels: int = 3):
        """
        Train a sentiment classification model.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_labels (int): Number of sentiment classes
        """
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        logger.info("Starting model training...")
        trainer.train()

        # Save the model
        trainer.save_model("./trained_sentiment_model")
        logger.info("Model training completed and saved")

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset class for PyTorch training."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)