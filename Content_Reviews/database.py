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
                    review_id TEXT NOT NULL,
                    is_spam BOOLEAN NOT NULL,
                    is_inappropriate BOOLEAN NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    flagged_for_review BOOLEAN DEFAULT FALSE,
                    processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
                )
            ''')

            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_review_id 
                ON moderation_results(review_id)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_flagged 
                ON moderation_results(flagged_for_review)
            ''')

            conn.commit()

    def insert_reviews_from_csv(self, df: pd.DataFrame) -> int:
        """
        Insert reviews from CSV DataFrame into database

        Args:
            df: DataFrame containing review data

        Returns:
            Number of reviews inserted
        """
        inserted_count = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO reviews 
                        (review_id, review_text, rating, date, reviewer_name)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        str(row['review_id']),
                        row['review_text'],
                        int(row['rating']),
                        row['date'],
                        row['reviewer_name']
                    ))

                    if cursor.rowcount > 0:
                        inserted_count += 1

                except Exception as e:
                    print(f"Error inserting review {row['review_id']}: {e}")
                    continue

            conn.commit()

        return inserted_count

    def insert_moderation_result(self, review_id: str, analysis: Dict[str, Any]) -> bool:
        """
        Insert moderation analysis result

        Args:
            review_id: ID of the review
            analysis: Dictionary containing moderation analysis

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if analysis already exists
                cursor.execute(
                    "SELECT id FROM moderation_results WHERE review_id = ?",
                    (review_id,)
                )

                if cursor.fetchone():
                    # Update existing record
                    cursor.execute('''
                        UPDATE moderation_results SET
                        is_spam = ?, is_inappropriate = ?, sentiment = ?,
                        confidence = ?, reasoning = ?, flagged_for_review = ?,
                        processed_at = CURRENT_TIMESTAMP
                        WHERE review_id = ?
                    ''', (
                        analysis['is_spam'],
                        analysis['is_inappropriate'],
                        analysis['sentiment'],
                        analysis['confidence'],
                        analysis['reasoning'],
                        analysis['is_spam'] or analysis['is_inappropriate'],
                        review_id
                    ))
                else:
                    # Insert new record
                    cursor.execute('''
                        INSERT INTO moderation_results 
                        (review_id, is_spam, is_inappropriate, sentiment, 
                         confidence, reasoning, flagged_for_review)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        review_id,
                        analysis['is_spam'],
                        analysis['is_inappropriate'],
                        analysis['sentiment'],
                        analysis['confidence'],
                        analysis['reasoning'],
                        analysis['is_spam'] or analysis['is_inappropriate']
                    ))

                conn.commit()
                return True

        except Exception as e:
            print(f"Error inserting moderation result: {e}")
            return False

    def get_all_reviews(self) -> pd.DataFrame:
        """Get all reviews as DataFrame"""
        query = '''
            SELECT r.review_id, r.review_text, r.rating, r.date, r.reviewer_name,
                   mr.is_spam, mr.is_inappropriate, mr.sentiment, mr.confidence,
                   mr.reasoning, mr.flagged_for_review
            FROM reviews r
            LEFT JOIN moderation_results mr ON r.review_id = mr.review_id
            ORDER BY r.date DESC
        '''

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_flagged_reviews(self) -> pd.DataFrame:
        """Get only flagged reviews"""
        query = '''
            SELECT r.review_id, r.review_text, r.rating, r.date, r.reviewer_name,
                   mr.is_spam, mr.is_inappropriate, mr.sentiment, mr.confidence,
                   mr.reasoning
            FROM reviews r
            JOIN moderation_results mr ON r.review_id = mr.review_id
            WHERE mr.flagged_for_review = 1
            ORDER BY mr.processed_at DESC
        '''

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_unprocessed_reviews(self) -> pd.DataFrame:
        """Get reviews that haven't been processed yet"""
        query = '''
            SELECT review_id, review_text, rating, date, reviewer_name
            FROM reviews r
            WHERE NOT EXISTS (
                SELECT 1 FROM moderation_results mr 
                WHERE mr.review_id = r.review_id
            )
            ORDER BY date DESC
        '''

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about reviews and moderation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM reviews")
            total_reviews = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM moderation_results")
            processed_reviews = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM moderation_results WHERE flagged_for_review = 1")
            flagged_reviews = cursor.fetchone()[0]

            # Sentiment distribution
            cursor.execute('''
                SELECT sentiment, COUNT(*) 
                FROM moderation_results 
                GROUP BY sentiment
            ''')
            sentiment_dist = dict(cursor.fetchall())

            # Rating distribution
            cursor.execute('''
                SELECT rating, COUNT(*) 
                FROM reviews 
                GROUP BY rating 
                ORDER BY rating
            ''')
            rating_dist = dict(cursor.fetchall())

            # Spam and inappropriate counts
            cursor.execute("SELECT COUNT(*) FROM moderation_results WHERE is_spam = 1")
            spam_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM moderation_results WHERE is_inappropriate = 1")
            inappropriate_count = cursor.fetchone()[0]

            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM moderation_results")
            avg_confidence = cursor.fetchone()[0] or 0

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