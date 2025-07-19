# database.py
"""
Database operations for the Content Moderation Tool
Handles SQLite database creation, data insertion, and queries
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import DATABASE_PATH


class DatabaseManager:
    """Manages all database operations for the content moderation system"""

    def __init__(self, db_path: str = DATABASE_PATH):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create reviews table - stores original review data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT UNIQUE NOT NULL,
                    review_text TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    reviewer_name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create moderation_results table - stores AI analysis results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS moderation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT UNIQUE NOT NULL,
                    is_spam BOOLEAN NOT NULL,
                    is_inappropriate BOOLEAN NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    analysis_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (review_id) REFERENCES reviews(review_id)
                )
            ''')
            conn.commit()

    def add_reviews_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Add new reviews from a DataFrame to the database, skipping duplicates.

        Args:
            df: DataFrame containing review data.

        Returns:
            Number of new rows added.
        """
        rows_added = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for index, row in df.iterrows():
                try:
                    cursor.execute(
                        "INSERT INTO reviews (review_id, review_text, rating, date, reviewer_name) VALUES (?, ?, ?, ?, ?)",
                        (str(row['review_id']), row['review_text'], int(row['rating']), row['date'], row['reviewer_name'])
                    )
                    rows_added += 1
                except sqlite3.IntegrityError:
                    # review_id already exists, skip
                    pass
            conn.commit()
        return rows_added

    def add_analysis_result(self, analysis: Dict[str, Any]):
        """
        Add AI analysis result to the database, updating if exists.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO moderation_results
                   (review_id, is_spam, is_inappropriate, sentiment, confidence, reasoning, analysis_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    analysis['review_id'],
                    analysis['is_spam'],
                    analysis['is_inappropriate'],
                    analysis['sentiment'],
                    analysis['confidence'],
                    analysis['reasoning'],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            )
            conn.commit()

    def get_unprocessed_reviews(self) -> List[Dict[str, Any]]:
        """
        Get reviews that have not yet been analyzed.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row # Allows access by column name
            cursor = conn.cursor()
            cursor.execute(
                """SELECT r.review_id, r.review_text, r.rating, r.date, r.reviewer_name
                   FROM reviews r
                   LEFT JOIN moderation_results mr ON r.review_id = mr.review_id
                   WHERE mr.review_id IS NULL"""
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_flagged_reviews(self) -> pd.DataFrame:
        """
        Get all reviews that have been flagged as spam or inappropriate.
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT r.review_id, r.review_text, r.rating, r.date, r.reviewer_name,
                       mr.is_spam, mr.is_inappropriate, mr.sentiment, mr.confidence, mr.reasoning
                FROM reviews r
                JOIN moderation_results mr ON r.review_id = mr.review_id
                WHERE mr.is_spam = 1 OR mr.is_inappropriate = 1
                ORDER BY mr.analysis_date DESC
            """
            return pd.read_sql_query(query, conn)

    def get_all_reviews_with_analysis(self) -> pd.DataFrame:
        """
        Get all reviews along with their analysis results.
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT r.review_id, r.review_text, r.rating, r.date, r.reviewer_name,
                       mr.is_spam, mr.is_inappropriate, mr.sentiment, mr.confidence, mr.reasoning,
                       mr.analysis_date
                FROM reviews r
                LEFT JOIN moderation_results mr ON r.review_id = mr.review_id
                ORDER BY r.date DESC
            """
            return pd.read_sql_query(query, conn)

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Retrieve various statistics for the dashboard.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            total_reviews = cursor.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
            processed_reviews = cursor.execute("SELECT COUNT(*) FROM moderation_results").fetchone()[0]
            flagged_reviews = cursor.execute("SELECT COUNT(*) FROM moderation_results WHERE is_spam = 1 OR is_inappropriate = 1").fetchone()[0]
            spam_count = cursor.execute("SELECT COUNT(*) FROM moderation_results WHERE is_spam = 1").fetchone()[0]
            inappropriate_count = cursor.execute("SELECT COUNT(*) FROM moderation_results WHERE is_inappropriate = 1").fetchone()[0]

            # Sentiment distribution
            sentiment_query = "SELECT sentiment, COUNT(*) FROM moderation_results GROUP BY sentiment"
            sentiment_results = cursor.execute(sentiment_query).fetchall()
            sentiment_dist = {row[0]: row[1] for row in sentiment_results}

            # Rating distribution
            rating_query = "SELECT rating, COUNT(*) FROM reviews GROUP BY rating"
            rating_results = cursor.execute(rating_query).fetchall()
            rating_dist = {row[0]: row[1] for row in rating_results}

            # Average confidence
            avg_confidence_result = cursor.execute("SELECT AVG(confidence) FROM moderation_results").fetchone()[0]
            avg_confidence = avg_confidence_result if avg_confidence_result is not None else 0.0


            return {
                'total_reviews': total_reviews,
                'processed_reviews': processed_reviews,
                'flagged_reviews': flagged_reviews,
                'unprocessed_reviews': total_reviews - processed_reviews,
                'spam_count': spam_count,
                'inappropriate_count': inappropriate_count,
                'sentiment_distribution': sentiment_dist,
                'rating_distribution': rating_dist,
                'average_confidence': round(avg_confidence, 2),
                'processing_rate': round((processed_reviews / total_reviews * 100), 1) if total_reviews > 0 else 0
            }

    def clear_all_data(self):
        """Clear all data from database (useful for testing)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM moderation_results")
            cursor.execute("DELETE FROM reviews")
            conn.commit()

    def get_reviewer_patterns(self) -> pd.DataFrame:
        """Analyze reviewer patterns for suspicious activity"""
        query = '''
            SELECT reviewer_name,
                   COUNT(*) as review_count,
                   AVG(rating) as avg_rating,
                   MIN(date) as first_review,
                   MAX(date) as last_review,
                   COUNT(DISTINCT date) as unique_dates
            FROM reviews
            GROUP BY reviewer_name
            HAVING review_count > 1
            ORDER BY review_count DESC
        '''

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)