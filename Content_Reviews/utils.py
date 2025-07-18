# utils.py
"""
Utility functions for the Content Moderation Tool
Handles CSV processing, data validation, formatting, and helper functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import re
import streamlit as st
from io import StringIO

from config import REQUIRED_COLUMNS, DATE_FORMAT, MAX_DISPLAY_LENGTH


class CSVProcessor:
    """Handles CSV file processing and validation"""

    @staticmethod
    def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that CSV has required columns and proper structure

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if DataFrame is empty
        if df.empty:
            errors.append("CSV file is empty")
            return False, errors

        # Check required columns
        missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

        # Check for extra columns (warning, not error)
        extra_columns = set(df.columns) - set(REQUIRED_COLUMNS)
        if extra_columns:
            st.warning(f"Extra columns found (will be ignored): {', '.join(extra_columns)}")

        # Validate data types and content
        if 'review_id' in df.columns:
            if df['review_id'].isnull().any():
                errors.append("review_id column contains null values")
            if df['review_id'].duplicated().any():
                duplicates = df['review_id'].duplicated().sum()
                errors.append(f"review_id column contains {duplicates} duplicate values")

        if 'review_text' in df.columns:
            null_reviews = df['review_text'].isnull().sum()
            if null_reviews > 0:
                errors.append(f"review_text column contains {null_reviews} null values")

            empty_reviews = (df['review_text'].str.strip() == '').sum()
            if empty_reviews > 0:
                errors.append(f"{empty_reviews} reviews have empty text")

        if 'rating' in df.columns:
            try:
                ratings = pd.to_numeric(df['rating'], errors='coerce')
                invalid_ratings = ratings.isnull().sum()
                if invalid_ratings > 0:
                    errors.append(f"{invalid_ratings} invalid rating values (must be numeric)")

                valid_ratings = ratings.dropna()
                if len(valid_ratings) > 0:
                    if (valid_ratings < 1).any() or (valid_ratings > 5).any():
                        errors.append("Rating values must be between 1 and 5")
            except:
                errors.append("Rating column contains non-numeric values")

        if 'date' in df.columns:
            invalid_dates = CSVProcessor._validate_dates(df['date'])
            if invalid_dates > 0:
                errors.append(f"{invalid_dates} invalid date values (expected format: YYYY-MM-DD)")

        if 'reviewer_name' in df.columns:
            null_names = df['reviewer_name'].isnull().sum()
            if null_names > 0:
                errors.append(f"reviewer_name column contains {null_names} null values")

        return len(errors) == 0, errors

    @staticmethod
    def _validate_dates(date_series: pd.Series) -> int:
        """
        Validate date format and count invalid dates

        Args:
            date_series: Series of date strings

        Returns:
            Number of invalid dates
        """
        invalid_count = 0

        for date_str in date_series:
            if pd.isnull(date_str):
                invalid_count += 1
                continue

            try:
                datetime.strptime(str(date_str), DATE_FORMAT)
            except ValueError:
                invalid_count += 1

        return invalid_count

    @staticmethod
    def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare CSV data for processing

        Args:
            df: Raw DataFrame from CSV

        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        clean_df = df.copy()

        # Keep only required columns
        clean_df = clean_df[REQUIRED_COLUMNS]

        # Clean review_text
        clean_df['review_text'] = clean_df['review_text'].astype(str).str.strip()

        # Clean and convert rating
        clean_df['rating'] = pd.to_numeric(clean_df['rating'], errors='coerce')

        # Clean reviewer_name
        clean_df['reviewer_name'] = clean_df['reviewer_name'].astype(str).str.strip()

        # Clean review_id
        clean_df['review_id'] = clean_df['review_id'].astype(str).str.strip()

        # Ensure date is string format
        clean_df['date'] = clean_df['date'].astype(str)

        # Remove rows with critical missing data
        clean_df = clean_df.dropna(subset=['review_id', 'review_text', 'rating'])

        # Remove duplicate review_ids, keeping first occurrence
        clean_df = clean_df.drop_duplicates(subset=['review_id'], keep='first')

        return clean_df

    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}

        # Basic stats
        total_reviews = len(df)

        # Date range
        try:
            dates = pd.to_datetime(df['date'], format=DATE_FORMAT)
            date_range = {
                'earliest': dates.min().strftime('%Y-%m-%d'),
                'latest': dates.max().strftime('%Y-%m-%d'),
                'span_days': (dates.max() - dates.min()).days
            }
        except:
            date_range = {'error': 'Could not parse dates'}

        # Rating distribution
        rating_dist = df['rating'].value_counts().sort_index().to_dict()

        # Text length statistics
        text_lengths = df['review_text'].str.len()

        # Reviewer statistics
        reviewer_stats = df['reviewer_name'].value_counts()

        return {
            'total_reviews': total_reviews,
            'date_range': date_range,
            'rating_distribution': rating_dist,
            'average_rating': round(df['rating'].mean(), 2),
            'text_length_stats': {
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'average': round(text_lengths.mean(), 1),
                'median': int(text_lengths.median())
            },
            'unique_reviewers': len(df['reviewer_name'].unique()),
            'reviews_per_reviewer': {
                'max': int(reviewer_stats.max()),
                'average': round(len(df) / len(reviewer_stats), 1)
            },
            'potential_duplicates': len(df) - len(df['reviewer_name'].unique())
        }


class DataFormatter:
    """Handles data formatting and display utilities"""

    @staticmethod
    def truncate_text(text: str, max_length: int = MAX_DISPLAY_LENGTH) -> str:
        """
        Truncate text for display purposes

        Args:
            text: Text to truncate
            max_length: Maximum length before truncation

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def format_confidence(confidence: float) -> str:
        """
        Format confidence score for display

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            Formatted confidence string
        """
        percentage = confidence * 100
        if percentage >= 80:
            return f"🟢 {percentage:.1f}%"
        elif percentage >= 60:
            return f"🟡 {percentage:.1f}%"
        else:
            return f"🔴 {percentage:.1f}%"

    @staticmethod
    def format_sentiment(sentiment: str) -> str:
        """
        Format sentiment with emoji

        Args:
            sentiment: Sentiment string

        Returns:
            Formatted sentiment with emoji
        """
        sentiment_map = {
            'positive': '😊 Positive',
            'negative': '😞 Negative',
            'neutral': '😐 Neutral'
        }
        return sentiment_map.get(sentiment.lower(), f"❓ {sentiment}")

    @staticmethod
    def format_boolean_flag(is_flagged: bool, flag_type: str) -> str:
        """
        Format boolean flags for display

        Args:
            is_flagged: Boolean flag value
            flag_type: Type of flag ('spam', 'inappropriate', etc.)

        Returns:
            Formatted flag string
        """
        if is_flagged:
            flag_map = {
                'spam': '🚫 Spam',
                'inappropriate': '⚠️ Inappropriate',
                'flagged': '🚩 Flagged'
            }
            return flag_map.get(flag_type.lower(), f"❌ {flag_type}")
        else:
            return f"✅ Clean"

    @staticmethod
    def create_summary_metrics(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create formatted metrics for Streamlit display

        Args:
            stats: Statistics dictionary

        Returns:
            List of metric dictionaries
        """
        metrics = []

        if 'total_reviews' in stats:
            metrics.append({
                'label': 'Total Reviews',
                'value': stats['total_reviews'],
                'delta': None
            })

        if 'flagged_reviews' in stats:
            flagged_pct = (stats['flagged_reviews'] / stats['total_reviews'] * 100) if stats['total_reviews'] > 0 else 0
            metrics.append({
                'label': 'Flagged Reviews',
                'value': stats['flagged_reviews'],
                'delta': f"{flagged_pct:.1f}%"
            })

        if 'average_confidence' in stats:
            metrics.append({
                'label': 'Avg Confidence',
                'value': f"{stats['average_confidence']:.2f}",
                'delta': None
            })

        if 'processing_rate' in stats:
            metrics.append({
                'label': 'Processing Rate',
                'value': f"{stats['processing_rate']:.1f}%",
                'delta': None
            })

        return metrics


class DateHelper:
    """Date-related utility functions"""

    @staticmethod
    def get_date_range_filter(df: pd.DataFrame, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame by date range

        Args:
            df: DataFrame to filter
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            Filtered DataFrame
        """
        if start_date is None and end_date is None:
            return df

        try:
            dates = pd.to_datetime(df['date'], format=DATE_FORMAT)
            mask = pd.Series([True] * len(df))

            if start_date:
                start_dt = datetime.strptime(start_date, DATE_FORMAT)
                mask &= (dates >= start_dt)

            if end_date:
                end_dt = datetime.strptime(end_date, DATE_FORMAT)
                mask &= (dates <= end_dt)

            return df[mask]
        except:
            st.error("Error filtering by date range")
            return df

    @staticmethod
    def get_recent_months(months: int = 6) -> Tuple[str, str]:
        """
        Get date range for recent months

        Args:
            months: Number of months to go back

        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30)  # Approximate

        return start_date.strftime(DATE_FORMAT), end_date.strftime(DATE_FORMAT)

    @staticmethod
    def parse_date_safely(date_str: str) -> Optional[datetime]:
        """
        Safely parse date string

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime or None if invalid
        """
        try:
            return datetime.strptime(str(date_str), DATE_FORMAT)
        except (ValueError, TypeError):
            return None


class ReviewAnalyzer:
    """Advanced review analysis utilities"""

    @staticmethod
    def detect_suspicious_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect suspicious patterns in review data

        Args:
            df: DataFrame with review data

        Returns:
            Dictionary with detected patterns
        """
        patterns = {}

        # Duplicate reviewer names
        reviewer_counts = df['reviewer_name'].value_counts()
        multiple_reviews = reviewer_counts[reviewer_counts > 1]
        patterns['multiple_reviewers'] = multiple_reviews.to_dict()

        # Reviews posted on same date
        date_counts = df['date'].value_counts()
        same_date_reviews = date_counts[date_counts > 3]  # More than 3 reviews on same date
        patterns['same_date_clusters'] = same_date_reviews.to_dict()

        # Rating patterns
        rating_dist = df['rating'].value_counts().sort_index()
        patterns['rating_distribution'] = rating_dist.to_dict()

        # Very short reviews with high ratings
        short_high_rated = df[(df['review_text'].str.len() < 20) & (df['rating'] >= 4)]
        patterns['short_high_rated_count'] = len(short_high_rated)

        # Very long reviews (potential spam)
        long_reviews = df[df['review_text'].str.len() > 500]
        patterns['long_reviews_count'] = len(long_reviews)

        return patterns

    @staticmethod
    def get_reviewer_insights(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get insights about reviewers

        Args:
            df: DataFrame with review data

        Returns:
            DataFrame with reviewer insights
        """
        reviewer_insights = df.groupby('reviewer_name').agg({
            'review_id': 'count',
            'rating': ['mean', 'std'],
            'date': ['min', 'max'],
            'review_text': lambda x: x.str.len().mean()
        }).round(2)

        reviewer_insights.columns = ['review_count', 'avg_rating', 'rating_std', 'first_review', 'last_review',
                                     'avg_text_length']

        # Add suspicion score
        reviewer_insights['suspicion_score'] = (
                (reviewer_insights['review_count'] > 3) * 0.3 +  # Multiple reviews
                (reviewer_insights['rating_std'] < 0.5) * 0.2 +  # Consistent ratings
                (reviewer_insights['avg_text_length'] < 30) * 0.3 +  # Short reviews
                (reviewer_insights['avg_rating'] == 5.0) * 0.2  # All 5-star reviews
        )

        return reviewer_insights.sort_values('suspicion_score', ascending=False)