import sqlite3
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Handles all database operations for the content moderation system.
    Uses SQLite for simplicity and local storage.
    """

    def __init__(self, db_path='content_moderation.db'):
        """
        Initialize database connection and create tables if they don't exist.

        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Create and return a database connection."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This allows accessing columns by name
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def init_database(self):
        """
        Create necessary tables if they don't exist.
        This runs when the DatabaseManager is first initialized.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Create reviews table - stores original review data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT UNIQUE,
                    review_text TEXT NOT NULL,
                    rating INTEGER,
                    date TEXT,
                    reviewer_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create predictions table - stores ML model predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    spam_probability REAL,
                    is_spam BOOLEAN,
                    toxicity_score REAL,
                    is_toxic BOOLEAN,
                    confidence_score REAL,
                    model_version TEXT,
                    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
                )
            ''')

            # Create moderation_queue table - for human review of flagged content
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS moderation_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT,
                    reason TEXT,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    assigned_to TEXT,
                    reviewed_at TIMESTAMP,
                    action_taken TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (review_id) REFERENCES reviews (review_id)
                )
            ''')

            conn.commit()
            logger.info("Database initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_review(self, review_data):
        """
        Insert a single review into the database.

        Args:
            review_data (dict): Dictionary containing review information

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO reviews 
                (review_id, review_text, rating, date, reviewer_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                review_data['review_id'],
                review_data['review_text'],
                review_data.get('rating'),
                review_data.get('date'),
                review_data.get('reviewer_name')
            ))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting review: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def bulk_insert_reviews(self, reviews_df):
        """
        Insert multiple reviews from a pandas DataFrame.

        Args:
            reviews_df (pd.DataFrame): DataFrame containing review data

        Returns:
            int: Number of successfully inserted reviews
        """
        conn = self.get_connection()
        try:
            # Use pandas to_sql for efficient bulk insertion
            reviews_df.to_sql('reviews', conn, if_exists='append', index=False)
            conn.commit()
            logger.info(f"Bulk inserted {len(reviews_df)} reviews")
            return len(reviews_df)
        except Exception as e:
            logger.error(f"Error bulk inserting reviews: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def insert_prediction(self, review_id, predictions):
        """
        Insert model predictions for a review.

        Args:
            review_id (str): ID of the review
            predictions (dict): Dictionary containing prediction results

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (review_id, sentiment_score, sentiment_label, spam_probability, 
                 is_spam, toxicity_score, is_toxic, confidence_score, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                review_id,
                predictions.get('sentiment_score'),
                predictions.get('sentiment_label'),
                predictions.get('spam_probability'),
                predictions.get('is_spam'),
                predictions.get('toxicity_score'),
                predictions.get('is_toxic'),
                predictions.get('confidence_score'),
                predictions.get('model_version', 'v1.0')
            ))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error inserting prediction: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_reviews_for_processing(self, limit=100):
        """
        Get reviews that haven't been processed yet.

        Args:
            limit (int): Maximum number of reviews to return

        Returns:
            list: List of review dictionaries
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.* FROM reviews r
                LEFT JOIN predictions p ON r.review_id = p.review_id
                WHERE p.review_id IS NULL
                LIMIT ?
            ''', (limit,))

            reviews = []
            for row in cursor.fetchall():
                reviews.append({
                    'review_id': row['review_id'],
                    'review_text': row['review_text'],
                    'rating': row['rating'],
                    'date': row['date'],
                    'reviewer_name': row['reviewer_name']
                })
            return reviews
        except sqlite3.Error as e:
            logger.error(f"Error fetching reviews for processing: {e}")
            return []
        finally:
            conn.close()

    def get_flagged_content(self, limit=50):
        """
        Get content that has been flagged for manual review.

        Args:
            limit (int): Maximum number of items to return

        Returns:
            list: List of flagged content
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.review_text, r.reviewer_name, r.date, 
                       p.sentiment_label, p.spam_probability, p.toxicity_score,
                       m.reason, m.priority, m.status
                FROM reviews r
                JOIN predictions p ON r.review_id = p.review_id
                JOIN moderation_queue m ON r.review_id = m.review_id
                WHERE m.status = 'pending'
                ORDER BY m.priority DESC, m.created_at ASC
                LIMIT ?
            ''', (limit,))

            flagged_items = []
            for row in cursor.fetchall():
                flagged_items.append({
                    'review_text': row['review_text'],
                    'reviewer_name': row['reviewer_name'],
                    'date': row['date'],
                    'sentiment_label': row['sentiment_label'],
                    'spam_probability': row['spam_probability'],
                    'toxicity_score': row['toxicity_score'],
                    'reason': row['reason'],
                    'priority': row['priority'],
                    'status': row['status']
                })
            return flagged_items
        except sqlite3.Error as e:
            logger.error(f"Error fetching flagged content: {e}")
            return []
        finally:
            conn.close()

    def add_to_moderation_queue(self, review_id, reason, priority=1):
        """
        Add content to the moderation queue for human review.

        Args:
            review_id (str): ID of the review
            reason (str): Reason for flagging
            priority (int): Priority level (1-5, higher = more urgent)

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO moderation_queue (review_id, reason, priority)
                VALUES (?, ?, ?)
            ''', (review_id, reason, priority))
            conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding to moderation queue: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_analytics_data(self):
        """
        Get analytics data for dashboard.

        Returns:
            dict: Dictionary containing analytics metrics
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Get total reviews processed
            cursor.execute('SELECT COUNT(*) as total FROM reviews')
            total_reviews = cursor.fetchone()['total']

            # Get sentiment distribution
            cursor.execute('''
                SELECT sentiment_label, COUNT(*) as count
                FROM predictions
                GROUP BY sentiment_label
            ''')
            sentiment_dist = {row['sentiment_label']: row['count'] for row in cursor.fetchall()}

            # Get spam detection stats
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END) as spam_count,
                    SUM(CASE WHEN is_spam = 0 THEN 1 ELSE 0 END) as clean_count
                FROM predictions
            ''')
            spam_stats = cursor.fetchone()

            # Get moderation queue stats
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM moderation_queue
                GROUP BY status
            ''')
            queue_stats = {row['status']: row['count'] for row in cursor.fetchall()}

            return {
                'total_reviews': total_reviews,
                'sentiment_distribution': sentiment_dist,
                'spam_detected': spam_stats['spam_count'] if spam_stats else 0,
                'clean_content': spam_stats['clean_count'] if spam_stats else 0,
                'moderation_queue': queue_stats
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting analytics data: {e}")
            return {}
        finally:
            conn.close()

    def close(self):
        """Close database connection (if needed for cleanup)."""
        # SQLite connections are closed automatically when the connection object is deleted
        pass