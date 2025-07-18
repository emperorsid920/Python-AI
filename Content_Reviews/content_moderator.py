# content_moderator.py
"""
Core content moderation logic using Ollama for AI analysis
Handles both AI-powered analysis and rule-based fallback
"""

import json
import requests
import re
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, CONFIDENCE_THRESHOLD,
    SPAM_KEYWORDS, INAPPROPRIATE_KEYWORDS, SUSPICIOUS_PATTERNS,
    POSITIVE_WORDS, NEGATIVE_WORDS
)


class ContentModerator:
    """Main class for content moderation using Ollama AI and rule-based fallback"""

    def __init__(self, ollama_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        """
        Initialize the content moderator

        Args:
            ollama_url: Base URL for Ollama API
            model: Name of the Ollama model to use
        """
        self.ollama_url = ollama_url
        self.model = model
        self.is_ollama_available = self._check_ollama_connection()

    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            print("⚠️  Ollama not available - using fallback analysis")
            return False

    def analyze_review(self, review_text: str, rating: int, reviewer_name: str) -> Dict[str, Any]:
        """
        Analyze a single review for spam, inappropriate content, and sentiment

        Args:
            review_text: The review content to analyze
            rating: Star rating (1-5)
            reviewer_name: Name of the reviewer

        Returns:
            Dictionary containing analysis results
        """
        if self.is_ollama_available:
            try:
                return self._analyze_with_ollama(review_text, rating, reviewer_name)
            except Exception as e:
                print(f"Ollama analysis failed: {e}")
                return self._analyze_with_rules(review_text, rating, reviewer_name)
        else:
            return self._analyze_with_rules(review_text, rating, reviewer_name)

    def _analyze_with_ollama(self, review_text: str, rating: int, reviewer_name: str) -> Dict[str, Any]:
        """
        Use Ollama AI for sophisticated content analysis

        Args:
            review_text: Review content
            rating: Star rating
            reviewer_name: Reviewer name

        Returns:
            Analysis results from AI
        """
        # Craft a specific prompt for repair shop reviews
        prompt = f"""
        You are analyzing a review for an electronic repair shop. Please analyze this review for content moderation:

        Review: "{review_text}"
        Rating: {rating}/5 stars
        Reviewer: {reviewer_name}

        Context: This is for an electronic repair shop that fixes phones, laptops, gaming consoles, etc.

        Analyze for:
        1. SPAM: Fake reviews, generic praise, competitor bashing, promotional content
        2. INAPPROPRIATE: Offensive language, personal attacks, threats, harassment
        3. SENTIMENT: Overall emotional tone considering the repair shop context

        Consider these red flags:
        - Rating-text mismatch (5 stars but negative text, or 1 star but positive text)
        - Overly generic language ("best service ever", "perfect job")
        - No mention of actual repair services
        - Suspicious reviewer patterns

        Respond ONLY with valid JSON in this exact format:
        {{
            "is_spam": true/false,
            "is_inappropriate": true/false,
            "sentiment": "positive/negative/neutral",
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation of your analysis focusing on key factors"
        }}
        """

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent analysis
                        "top_p": 0.9
                    }
                },
                timeout=OLLAMA_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()

                # Parse the AI response
                try:
                    analysis = json.loads(result['response'])

                    # Validate the response structure
                    required_keys = ['is_spam', 'is_inappropriate', 'sentiment', 'confidence', 'reasoning']
                    if all(key in analysis for key in required_keys):
                        # Add additional context-specific checks
                        analysis = self._enhance_analysis(analysis, review_text, rating, reviewer_name)
                        return analysis
                    else:
                        raise ValueError("Invalid response structure from AI")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error parsing AI response: {e}")
                    return self._analyze_with_rules(review_text, rating, reviewer_name)
            else:
                raise requests.RequestException(f"HTTP {response.status_code}")

        except requests.RequestException as e:
            print(f"Ollama API error: {e}")
            return self._analyze_with_rules(review_text, rating, reviewer_name)

    def _enhance_analysis(self, analysis: Dict[str, Any], review_text: str, rating: int, reviewer_name: str) -> Dict[
        str, Any]:
        """
        Enhance AI analysis with additional rule-based checks specific to repair shops

        Args:
            analysis: Original AI analysis
            review_text: Review text
            rating: Star rating
            reviewer_name: Reviewer name

        Returns:
            Enhanced analysis
        """
        # Check for rating-sentiment mismatch
        if rating >= 4 and analysis['sentiment'] == 'negative':
            analysis['is_spam'] = True
            analysis['reasoning'] += " | Rating-sentiment mismatch detected"

        if rating <= 2 and analysis['sentiment'] == 'positive':
            analysis['is_spam'] = True
            analysis['reasoning'] += " | Suspicious positive sentiment with low rating"

        # Check for overly generic content
        generic_phrases = SUSPICIOUS_PATTERNS['generic_positive_phrases']
        if any(phrase in review_text.lower() for phrase in generic_phrases):
            if len(review_text) < 50:  # Short and generic = likely spam
                analysis['is_spam'] = True
                analysis['reasoning'] += " | Generic short review detected"

        # Check for competitor mentions
        competitor_mentions = SUSPICIOUS_PATTERNS['competitor_mentions']
        if any(mention in review_text.lower() for mention in competitor_mentions):
            analysis['is_inappropriate'] = True
            analysis['reasoning'] += " | Competitor mention detected"

        # Check review length
        if len(review_text) < SUSPICIOUS_PATTERNS['min_review_length']:
            analysis['confidence'] = max(0.3, analysis['confidence'] - 0.2)
            analysis['reasoning'] += " | Very short review"

        return analysis

    def _analyze_with_rules(self, review_text: str, rating: int, reviewer_name: str) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when Ollama is not available

        Args:
            review_text: Review content
            rating: Star rating
            reviewer_name: Reviewer name

        Returns:
            Rule-based analysis results
        """
        review_lower = review_text.lower()

        # Spam detection
        is_spam = False
        spam_reasons = []

        # Check for spam keywords
        for keyword in SPAM_KEYWORDS:
            if keyword in review_lower:
                is_spam = True
                spam_reasons.append(f"spam keyword: {keyword}")

        # Check for generic positive phrases with high rating
        generic_phrases = SUSPICIOUS_PATTERNS['generic_positive_phrases']
        if rating >= 4 and any(phrase in review_lower for phrase in generic_phrases):
            if len(review_text) < 50:
                is_spam = True
                spam_reasons.append("generic short positive review")

        # Check for rating-length mismatch
        if rating == 5 and len(review_text) < 20:
            is_spam = True
            spam_reasons.append("suspiciously short 5-star review")

        # Inappropriate content detection
        is_inappropriate = False
        inappropriate_reasons = []

        for keyword in INAPPROPRIATE_KEYWORDS:
            if keyword in review_lower:
                is_inappropriate = True
                inappropriate_reasons.append(f"inappropriate keyword: {keyword}")

        # Check for competitor mentions
        competitor_mentions = SUSPICIOUS_PATTERNS['competitor_mentions']
        for mention in competitor_mentions:
            if mention in review_lower:
                is_inappropriate = True
                inappropriate_reasons.append("competitor mention")

        # Sentiment analysis
        sentiment = self._analyze_sentiment_rules(review_text)

        # Build reasoning
        reasoning_parts = []
        if spam_reasons:
            reasoning_parts.append(f"Spam indicators: {', '.join(spam_reasons)}")
        if inappropriate_reasons:
            reasoning_parts.append(f"Inappropriate content: {', '.join(inappropriate_reasons)}")
        if not spam_reasons and not inappropriate_reasons:
            reasoning_parts.append("No major issues detected")

        reasoning = " | ".join(reasoning_parts) + " (Rule-based analysis)"

        return {
            'is_spam': is_spam,
            'is_inappropriate': is_inappropriate,
            'sentiment': sentiment,
            'confidence': 0.6,  # Lower confidence for rule-based analysis
            'reasoning': reasoning
        }

    def _analyze_sentiment_rules(self, text: str) -> str:
        """
        Simple rule-based sentiment analysis

        Args:
            text: Text to analyze

        Returns:
            Sentiment: 'positive', 'negative', or 'neutral'
        """
        text_lower = text.lower()

        positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def batch_analyze(self, reviews_df: pd.DataFrame, progress_callback=None) -> List[Dict[str, Any]]:
        """
        Analyze multiple reviews in batch

        Args:
            reviews_df: DataFrame containing reviews to analyze
            progress_callback: Optional callback function for progress updates

        Returns:
            List of analysis results
        """
        results = []
        total_reviews = len(reviews_df)

        for index, row in reviews_df.iterrows():
            try:
                analysis = self.analyze_review(
                    row['review_text'],
                    row['rating'],
                    row['reviewer_name']
                )

                # Add review_id to the result
                analysis['review_id'] = str(row['review_id'])
                results.append(analysis)

                # Call progress callback if provided
                if progress_callback:
                    progress = (index + 1) / total_reviews
                    progress_callback(progress, index + 1, total_reviews)

            except Exception as e:
                print(f"Error analyzing review {row['review_id']}: {e}")
                # Add a failed analysis result
                results.append({
                    'review_id': str(row['review_id']),
                    'is_spam': False,
                    'is_inappropriate': False,
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'reasoning': f"Analysis failed: {str(e)}"
                })

        return results

    def get_analysis_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results

        Args:
            results: List of analysis results

        Returns:
            Summary statistics
        """
        if not results:
            return {}

        total_count = len(results)
        spam_count = sum(1 for r in results if r['is_spam'])
        inappropriate_count = sum(1 for r in results if r['is_inappropriate'])
        flagged_count = sum(1 for r in results if r['is_spam'] or r['is_inappropriate'])

        sentiment_counts = {}
        for r in results:
            sentiment = r['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

        avg_confidence = sum(r['confidence'] for r in results) / total_count

        return {
            'total_analyzed': total_count,
            'spam_detected': spam_count,
            'inappropriate_detected': inappropriate_count,
            'total_flagged': flagged_count,
            'flagged_percentage': round((flagged_count / total_count) * 100, 1),
            'sentiment_distribution': sentiment_counts,
            'average_confidence': round(avg_confidence, 2),
            'analysis_method': 'AI-powered' if self.is_ollama_available else 'Rule-based'
        }