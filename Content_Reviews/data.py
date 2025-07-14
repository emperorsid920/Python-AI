import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import string
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, cleaning, preprocessing, and feature engineering
    for the content moderation system.
    """

    def __init__(self):
        """Initialize the data processor with common preprocessing patterns."""
        # Common patterns for text cleaning
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{3}-?[0-9]{3}-?[0-9]{4}')
        self.mention_pattern = re.compile(r'@[\w]+')
        self.hashtag_pattern = re.compile(r'#[\w]+')

        # Initialize label encoder for sentiment labels
        self.label_encoder = LabelEncoder()

        # Common spam keywords (you can expand this list)
        self.spam_keywords = [
            'free', 'win', 'winner', 'urgent', 'limited time', 'act now',
            'click here', 'earn money', 'make money', 'work from home',
            'guarantee', 'risk free', 'no risk', 'call now', 'viagra',
            'weight loss', 'lose weight', 'miracle', 'amazing', 'incredible'
        ]

        logger.info("DataProcessor initialized")

    def load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load review data from CSV file.

        Args:
            file_path (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded and validated DataFrame
        """
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")

            # Validate required columns
            required_columns = ['review_id', 'review_text', 'rating', 'date', 'reviewer_name']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Basic data validation
            df = self._validate_data(df)

            logger.info(f"Data validation completed. Final dataset: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded data.

        Args:
            df (pd.DataFrame): Raw DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        initial_size = len(df)

        # Remove rows with missing review_text or review_id
        df = df.dropna(subset=['review_id', 'review_text'])

        # Remove duplicates based on review_id
        df = df.drop_duplicates(subset=['review_id'])

        # Clean review_text (remove empty strings, whitespace)
        df['review_text'] = df['review_text'].astype(str).str.strip()
        df = df[df['review_text'].str.len() > 0]

        # Validate rating column (should be numeric)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Clean date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Clean reviewer_name
        df['reviewer_name'] = df['reviewer_name'].astype(str).str.strip()

        logger.info(
            f"Data validation: {initial_size} -> {len(df)} rows (removed {initial_size - len(df)} invalid rows)")
        return df

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace URLs with placeholder
        text = self.url_pattern.sub(' [URL] ', text)

        # Replace emails with placeholder
        text = self.email_pattern.sub(' [EMAIL] ', text)

        # Replace phone numbers with placeholder
        text = self.phone_pattern.sub(' [PHONE] ', text)

        # Replace mentions with placeholder
        text = self.mention_pattern.sub(' [MENTION] ', text)

        # Replace hashtags with placeholder
        text = self.hashtag_pattern.sub(' [HASHTAG] ', text)

        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract additional features from the review data.

        Args:
            df (pd.DataFrame): DataFrame with review data

        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        df_features = df.copy()

        # Text length features
        df_features['text_length'] = df_features['review_text'].str.len()
        df_features['word_count'] = df_features['review_text'].str.split().str.len()
        df_features['sentence_count'] = df_features['review_text'].str.count(r'[.!?]+')

        # Clean text for analysis
        df_features['cleaned_text'] = df_features['review_text'].apply(self.clean_text)

        # Capitalization features
        df_features['caps_ratio'] = df_features['review_text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )

        # Punctuation features
        df_features['punct_ratio'] = df_features['review_text'].apply(
            lambda x: sum(1 for c in x if c in string.punctuation) / len(x) if len(x) > 0 else 0
        )

        # Exclamation and question marks
        df_features['exclamation_count'] = df_features['review_text'].str.count('!')
        df_features['question_count'] = df_features['review_text'].str.count('\\?')

        # URL and email indicators
        df_features['has_url'] = df_features['review_text'].str.contains(self.url_pattern, na=False)
        df_features['has_email'] = df_features['review_text'].str.contains(self.email_pattern, na=False)
        df_features['has_phone'] = df_features['review_text'].str.contains(self.phone_pattern, na=False)

        # Spam indicator features
        df_features['spam_keyword_count'] = df_features['cleaned_text'].apply(
            lambda x: sum(1 for keyword in self.spam_keywords if keyword in x.lower())
        )

        # Rating-based features
        df_features['is_extreme_rating'] = df_features['rating'].apply(
            lambda x: x in [1, 5] if pd.notna(x) else False
        )

        # Time-based features (if date is available)
        if 'date' in df_features.columns:
            df_features['date'] = pd.to_datetime(df_features['date'], errors='coerce')
            df_features['is_weekend'] = df_features['date'].dt.dayofweek.isin([5, 6])
            df_features['hour'] = df_features['date'].dt.hour
            df_features['is_night_post'] = df_features['hour'].between(22, 6)

        # Reviewer name features
        df_features['reviewer_name_length'] = df_features['reviewer_name'].str.len()
        df_features['has_numbers_in_name'] = df_features['reviewer_name'].str.contains(r'\d', na=False)

        logger.info(f"Feature extraction completed. Added {len(df_features.columns) - len(df.columns)} new features")
        return df_features

    def create_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment labels based on ratings for training data.

        Args:
            df (pd.DataFrame): DataFrame with rating column

        Returns:
            pd.DataFrame: DataFrame with sentiment labels
        """
        df_labeled = df.copy()

        # Create sentiment labels based on rating
        def rating_to_sentiment(rating):
            if pd.isna(rating):
                return 'neutral'
            elif rating <= 2:
                return 'negative'
            elif rating >= 4:
                return 'positive'
            else:
                return 'neutral'

        df_labeled['sentiment_label'] = df_labeled['rating'].apply(rating_to_sentiment)

        # Encode labels for ML models
        df_labeled['sentiment_encoded'] = self.label_encoder.fit_transform(df_labeled['sentiment_label'])

        logger.info("Sentiment labels created based on ratings")
        return df_labeled

    def detect_potential_spam(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spam indicators based on heuristics.

        Args:
            df (pd.DataFrame): DataFrame with features

        Returns:
            pd.DataFrame: DataFrame with spam indicators
        """
        df_spam = df.copy()

        # Spam score based on multiple factors
        spam_score = 0

        # High spam keyword count
        spam_score += (df_spam['spam_keyword_count'] > 2).astype(int) * 0.3

        # High caps ratio
        spam_score += (df_spam['caps_ratio'] > 0.3).astype(int) * 0.2

        # High exclamation marks
        spam_score += (df_spam['exclamation_count'] > 3).astype(int) * 0.2

        # Contains URLs or emails
        spam_score += df_spam['has_url'].astype(int) * 0.1
        spam_score += df_spam['has_email'].astype(int) * 0.1

        # Very short or very long text
        spam_score += ((df_spam['text_length'] < 20) | (df_spam['text_length'] > 1000)).astype(int) * 0.1

        df_spam['spam_score'] = spam_score
        df_spam['is_potential_spam'] = spam_score > 0.5

        logger.info(f"Spam detection completed. Found {df_spam['is_potential_spam'].sum()} potential spam reviews")
        return df_spam

    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training and validation datasets.

        Args:
            df (pd.DataFrame): Processed DataFrame
            test_size (float): Proportion of data to use for testing

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames
        """
        # Ensure we have necessary columns
        if 'sentiment_label' not in df.columns:
            df = self.create_sentiment_labels(df)

        # Split the data
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['sentiment_label'],
            random_state=42
        )

        logger.info(f"Training data prepared: {len(train_df)} train, {len(val_df)} validation")
        return train_df, val_df

    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive statistics about the dataset.

        Args:
            df (pd.DataFrame): DataFrame to analyze

        Returns:
            Dict: Dictionary containing various statistics
        """
        stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }

        # Text statistics
        if 'review_text' in df.columns:
            stats['text_stats'] = {
                'avg_length': df['review_text'].str.len().mean(),
                'max_length': df['review_text'].str.len().max(),
                'min_length': df['review_text'].str.len().min(),
                'avg_words': df['review_text'].str.split().str.len().mean()
            }

        # Rating statistics
        if 'rating' in df.columns:
            stats['rating_stats'] = {
                'mean_rating': df['rating'].mean(),
                'rating_distribution': df['rating'].value_counts().to_dict()
            }

        # Sentiment statistics (if available)
        if 'sentiment_label' in df.columns:
            stats['sentiment_distribution'] = df['sentiment_label'].value_counts().to_dict()

        # Spam statistics (if available)
        if 'is_potential_spam' in df.columns:
            stats['spam_stats'] = {
                'spam_count': df['is_potential_spam'].sum(),
                'spam_percentage': (df['is_potential_spam'].sum() / len(df)) * 100
            }

        logger.info("Data statistics computed")
        return stats

    def batch_process_reviews(self, reviews: List[Dict], batch_size: int = 100) -> List[Dict]:
        """
        Process reviews in batches for memory efficiency.

        Args:
            reviews (List[Dict]): List of review dictionaries
            batch_size (int): Number of reviews to process at once

        Returns:
            List[Dict]: List of processed reviews
        """
        processed_reviews = []

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            batch_df = pd.DataFrame(batch)

            # Process the batch
            batch_df = self.extract_features(batch_df)
            batch_df = self.detect_potential_spam(batch_df)

            # Convert back to list of dictionaries
            processed_batch = batch_df.to_dict('records')
            processed_reviews.extend(processed_batch)

            logger.info(f"Processed batch {i // batch_size + 1}/{len(reviews) // batch_size + 1}")

        return processed_reviews

    def export_processed_data(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        Export processed data to CSV.

        Args:
            df (pd.DataFrame): DataFrame to export
            output_path (str): Path to save the file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Data exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False