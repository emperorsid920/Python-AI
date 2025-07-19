# content_moderator.py
"""
Core content moderation logic using Gemini API for AI analysis
Handles both AI-powered analysis and rule-based fallback
"""

import json
import requests # Kept if other modules use it, otherwise can be removed
import re
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import os # Import os to get environment variables

# Import the Google Generative AI library
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config import (
    CONFIDENCE_THRESHOLD, SPAM_KEYWORDS, INAPPROPRIATE_KEYWORDS, SUSPICIOUS_PATTERNS,
    POSITIVE_WORDS, NEGATIVE_WORDS, GEMINI_API_KEY_ENV_VAR, GEMINI_MODEL, GEMINI_TIMEOUT
)


class ContentModerator:
    """Main class for content moderation using Gemini AI and rule-based fallback"""

    def __init__(self):
        """
        Initialize the content moderator
        """
        # Configure Gemini API
        api_key = os.getenv(GEMINI_API_KEY_ENV_VAR)
        if not api_key:
            print(f"❌ Error: Gemini API key not found in environment variable '{GEMINI_API_KEY_ENV_VAR}'.")
            print("Please set the GOOGLE_API_KEY environment variable or equivalent.")
            self.is_ai_available = False
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL)
            self.is_ai_available = self._check_gemini_connection()

    def _check_gemini_connection(self) -> bool:
        """Check if Gemini API is configured and callable."""
        try:
            # Attempt a small, non-resource intensive call to check connectivity
            # Listing models doesn't consume quota, but confirms connectivity
            for m in genai.list_models():
                if GEMINI_MODEL in m.name:
                    print(f"✅ Gemini model '{GEMINI_MODEL}' available.")
                    return True
            print(f"⚠️ Gemini model '{GEMINI_MODEL}' not found or not accessible.")
            return False
        except Exception as e:
            print(f"⚠️  Gemini API not available - using fallback analysis. Error: {e}")
            return False

    def analyze_review(self, review_text: str, rating: int, review_id: str) -> Dict[str, Any]:
        """
        Analyze a single review using AI (Gemini) or rule-based fallback.
        """
        analysis_result = {
            'review_id': review_id,
            'is_spam': False,
            'is_inappropriate': False,
            'sentiment': 'neutral',
            'confidence': 0.0,
            'reasoning': 'Rule-based analysis applied.'
        }

        if self.is_ai_available:
            try:
                # Construct the prompt for Gemini
                prompt = f"""
                Analyze the following electronic repair shop review for sentiment, potential spam, and inappropriate content.
                The review is: "{review_text}"
                The customer gave a rating of: {rating} out of 5.

                Based on the review content and rating, provide a JSON response with the following keys:
                - "sentiment": "positive", "negative", or "neutral"
                - "is_spam": true or false
                - "is_inappropriate": true or false
                - "confidence": a float between 0.0 and 1.0 indicating overall confidence in the analysis
                - "reasoning": a concise explanation for the sentiment, spam, or inappropriate flags.

                Consider the rating in your sentiment analysis.
                For spam, look for promotional language, repeated phrases, or clear attempts to mislead.
                For inappropriate content, look for hateful, abusive, or extremely offensive language.
                """

                response = self.model.generate_content(
                    prompt,
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE, # Added for completeness
                    },
                    request_options={"timeout": GEMINI_TIMEOUT}
                )

                # Assuming the response is text containing JSON
                ai_response_text = response.text.strip()
                # Attempt to parse only the JSON part if the model adds other text (e.g., in markdown block)
                json_match = re.search(r'```json\n({.*})\n```', ai_response_text, re.DOTALL)
                if json_match:
                    ai_analysis = json.loads(json_match.group(1))
                else:
                    # If no JSON block, try direct parse (some models might just output raw JSON)
                    ai_analysis = json.loads(ai_response_text)

                # Update analysis_result based on AI output
                analysis_result['sentiment'] = ai_analysis.get('sentiment', 'neutral').lower()
                analysis_result['is_spam'] = ai_analysis.get('is_spam', False)
                analysis_result['is_inappropriate'] = ai_analysis.get('is_inappropriate', False)
                analysis_result['confidence'] = float(ai_analysis.get('confidence', 0.0))
                analysis_result['reasoning'] = ai_analysis.get('reasoning', 'AI analysis performed.')

                # Apply confidence threshold: if AI confidence is low, revert to rule-based for flags
                if analysis_result['confidence'] < CONFIDENCE_THRESHOLD:
                    analysis_result['reasoning'] += " (AI confidence too low, falling back for flags)."
                    analysis_result['is_spam'] = self._check_spam_keywords(review_text) or self._check_suspicious_patterns(review_text)
                    analysis_result['is_inappropriate'] = self._check_inappropriate_keywords(review_text)
                    # Sentiment can still be taken from AI even with lower confidence if no strong rule applies.
                else:
                     analysis_result['reasoning'] = "AI analysis performed: " + analysis_result['reasoning']

            except Exception as e:
                print(f"❌ Gemini AI analysis failed for review ID {review_id}: {e}")
                analysis_result['reasoning'] = f"AI analysis failed: {str(e)}. Falling back to rule-based."
                # Fallback to rule-based if AI fails
                analysis_result['is_spam'] = self._check_spam_keywords(review_text) or self._check_suspicious_patterns(review_text)
                analysis_result['is_inappropriate'] = self._check_inappropriate_keywords(review_text)
                analysis_result['sentiment'] = self._rule_based_sentiment(review_text, rating)

        else:
            # Fallback to rule-based analysis if AI is not available
            analysis_result['is_spam'] = self._check_spam_keywords(review_text) or self._check_suspicious_patterns(review_text)
            analysis_result['is_inappropriate'] = self._check_inappropriate_keywords(review_text)
            analysis_result['sentiment'] = self._rule_based_sentiment(review_text, rating)
            analysis_result['reasoning'] = "Rule-based analysis applied due to AI unavailability."

        return analysis_result

    def get_analysis_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from analysis results
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

        avg_confidence = sum(r['confidence'] for r in results) / total_count if total_count > 0 else 0.0

        return {
            'total_analyzed': total_count,
            'spam_detected': spam_count,
            'inappropriate_detected': inappropriate_count,
            'total_flagged': flagged_count,
            'flagged_percentage': round((flagged_count / total_count) * 100, 1) if total_count > 0 else 0.0,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': round(avg_confidence, 2),
            'analysis_method': 'AI-powered' if self.is_ai_available else 'Rule-based Fallback' # Changed for Gemini
        }

    # --- Rule-based fallback methods (These remain the same) ---
    def _check_spam_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in SPAM_KEYWORDS)

    def _check_inappropriate_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in INAPPROPRIATE_KEYWORDS)

    def _check_suspicious_patterns(self, text: str) -> bool:
        text_lower = text.lower()
        if len(text_lower) < SUSPICIOUS_PATTERNS['min_review_length'] or \
           len(text_lower) > SUSPICIOUS_PATTERNS['max_review_length']:
            return True
        if any(phrase in text_lower for phrase in SUSPICIOUS_PATTERNS['generic_positive_phrases']):
            return True
        if any(mention in text_lower for mention in SUSPICIOUS_PATTERNS['competitor_mentions']):
            return True
        return False

    def _rule_based_sentiment(self, text: str, rating: int) -> str:
        text_lower = text.lower()
        pos_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
        neg_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)

        if rating >= 4: # Strong positive indicator from rating
            if neg_count == 0 or pos_count > neg_count:
                return 'positive'
        elif rating <= 2: # Strong negative indicator from rating
            if pos_count == 0 or neg_count > pos_count:
                return 'negative'

        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral' # Default if no strong signal