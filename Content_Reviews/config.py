# config.py
"""
Configuration settings for the Content Moderation Tool
This file centralizes all configuration parameters to make them easy to modify
"""

import os

# --- Gemini Configuration ---
# Set the environment variable GOOGLE_API_KEY with your actual Gemini API key
GEMINI_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
GEMINI_MODEL = "gemini-1.5-flash"  # Using Gemini 1.5 Flash
GEMINI_TIMEOUT = 60  # seconds to wait for Gemini response

# --- Database Configuration ---
DATABASE_PATH = "content_moderation.db"

# --- Moderation Thresholds ---
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to trust AI analysis
SPAM_KEYWORDS = [
    'buy now', 'click here', 'limited time', 'act now', 'free money',
    'guaranteed', 'no questions asked', 'risk free', 'call now'
]

INAPPROPRIATE_KEYWORDS = [
    'hate', 'stupid', 'idiot', 'scam', 'fraud', 'terrible',
    'worst', 'horrible', 'awful', 'disgusting'
]

# --- Review-specific Configuration ---
SUSPICIOUS_PATTERNS = {
    'min_review_length': 10,  # Flag reviews shorter than this
    'max_review_length': 1000,  # Flag reviews longer than this
    'generic_positive_phrases': [
        'flawless service', 'top notch', 'highly recommend',
        'best service ever', 'amazing work', 'perfect job'
    ],
    'competitor_mentions': [
        'go elsewhere', 'try another', 'better options',
        'competitor', 'other shops'
    ]
}

# --- Sentiment Analysis ---
POSITIVE_WORDS = [
    'excellent', 'great', 'amazing', 'fantastic', 'wonderful',
    'perfect', 'outstanding', 'superb', 'brilliant', 'awesome',
    'professional', 'friendly', 'helpful', 'quick', 'efficient'
]

NEGATIVE_WORDS = [
    'terrible', 'awful', 'horrible', 'bad', 'worst',
    'disappointed', 'frustrated', 'angry', 'upset', 'unsatisfied',
    'slow', 'expensive', 'rude', 'unprofessional', 'broken'
]

# --- Streamlit Configuration ---
PAGE_TITLE = "Electronic Repair Shop Review Moderator"
PAGE_ICON = "🔧"
LAYOUT = "wide"

# --- CSV Processing Configuration ---
REQUIRED_COLUMNS = ['review_id', 'review_text', 'rating', 'date', 'reviewer_name']
DATE_FORMAT = '%Y-%m-%d' # Expected date format in CSV
REVIEWS_PER_PAGE = 10 # For pagination in UI
MAX_DISPLAY_LENGTH = 150 # Max characters to display for review text snippets