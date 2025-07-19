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
import io # <--- ADD THIS LINE HERE

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
            # This check is now robust because we're forcing dtype in load_csv
            if not pd.api.types.is_string_dtype(df['review_id']):
                errors.append("'review_id' column must be of string type.")
            if df['review_id'].duplicated().any():
                errors.append("Duplicate 'review_id' values found. Review IDs must be unique.")


        if 'review_text' in df.columns:
            if not pd.api.types.is_string_dtype(df['review_text']):
                errors.append("'review_text' column must be of string type.")

        if 'rating' in df.columns:
            if not pd.api.types.is_integer_dtype(df['rating']):
                errors.append("'rating' column must be of integer type.")
            if not ((df['rating'] >= 1) & (df['rating'] <= 5)).all():
                errors.append("'rating' values must be between 1 and 5.")

        if 'date' in df.columns:
            try:
                # Attempt to convert to datetime to validate format
                pd.to_datetime(df['date'], format=DATE_FORMAT)
            except ValueError:
                errors.append(f"'date' column must be in '{DATE_FORMAT}' format.")

        if 'reviewer_name' in df.columns:
            if not pd.api.types.is_string_dtype(df['reviewer_name']):
                errors.append("'reviewer_name' column must be of string type.")

        if not errors:
            return True, []
        else:
            return False, errors

    def load_csv(self, uploaded_file: io.BytesIO) -> Optional[pd.DataFrame]:
        """
        Load a CSV file into a pandas DataFrame.
        Explicitly set 'review_id' to be read as string to prevent type issues.

        Args:
            uploaded_file: The file-like object uploaded by Streamlit.

        Returns:
            A pandas DataFrame if successful, None otherwise.
        """
        try:
            # Add dtype={'review_id': str} to force it to be read as a string
            df = pd.read_csv(uploaded_file, dtype={'review_id': str})
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None

    @staticmethod
    def get_common_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze common patterns in reviews (e.g., short highly-rated, long reviews)

        Args:
            df: DataFrame with review data

        Returns:
            Dictionary with pattern statistics
        """
        patterns = {}

        # Short, highly-rated reviews (potential fake positive)
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
                (reviewer_insights['avg_text_length'] > 500) * 0.2 # Very long reviews (also suspicious)
        )
        reviewer_insights['suspicion_score'] = reviewer_insights['suspicion_score'].apply(lambda x: min(x, 1.0)) # Cap at 1.0

        return reviewer_insights.sort_values('suspicion_score', ascending=False)


# --- Formatting Helper Functions ---
def format_confidence(confidence: float) -> str:
    """Formats a confidence score as a percentage string."""
    return f"{confidence:.1%}"

def format_date(date_str: str) -> str:
    """Formats a date string for display."""
    try:
        dt_obj = datetime.strptime(date_str, DATE_FORMAT)
        return dt_obj.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return date_str # Return as is if formatting fails

def truncate_text(text: str, max_length: int) -> str:
    """Truncates text to a max_length, adding '...' if truncated."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text