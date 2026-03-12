"""
Intent Classification for Appointment Conversations
Uses DeBERTa v3 zero-shot classification model (local) + Gemini API fallback
Processes both chatX.csv and screenshot_conversations.csv
"""

import pandas as pd
from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import os
import warnings
import json
import time

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


def load_data(csv_path, screenshot_csv_path=None):
    """Load data from chatX.csv and optionally screenshot_conversations.csv"""

    all_data = []

    # Load chatX.csv
    print(f"Loading {csv_path}...")
    df_chat = pd.read_csv(csv_path)
    df_chat = df_chat[df_chat['Sentences'].notna()]

    for idx, row in df_chat.iterrows():
        all_data.append({
            'Source': 'chatX.csv',
            'Index': idx + 1,
            'Text': row['Sentences']
        })

    print(f"  ✓ Loaded {len(df_chat)} conversations from chatX.csv")

    # Load screenshot_conversations.csv if provided
    if screenshot_csv_path and os.path.exists(screenshot_csv_path):
        print(f"Loading {screenshot_csv_path}...")
        df_screenshots = pd.read_csv(screenshot_csv_path)
        df_screenshots = df_screenshots[df_screenshots['Sentences'].notna()]

        for idx, row in df_screenshots.iterrows():
            all_data.append({
                'Source': 'screenshots',
                'Index': idx + 1,
                'Text': row['Sentences']
            })

        print(f"  ✓ Loaded {len(df_screenshots)} conversations from screenshots")

    print(f"\n✓ Total conversations to classify: {len(all_data)}\n")

    return pd.DataFrame(all_data)


def initialize_classifier():
    """Initialize the zero-shot classification pipeline"""
    print("Loading DeBERTa v3 zero-shot classification model...")
    print("This may take a moment on first run as it downloads the model...\n")

    import torch

    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        device=-1,  # Use CPU (-1), change to 0 for GPU if available
        framework="pt",  # Explicitly use PyTorch to avoid TensorFlow/Keras issues
        torch_dtype=torch.float32  # Force full precision to avoid Half precision error on CPU
    )
    return classifier


def initialize_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key or api_key == 'your_gemini_api_key_here':
        print("⚠️  Gemini API key not found in .env file")
        print("   Hybrid mode disabled - will use local model only")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    print("✓ Gemini API initialized for hybrid mode\n")
    return model


def classify_with_local_model(classifier, text, intent_labels, threshold=0.5):
    """Classify using local DeBERTa model"""
    result = classifier(text, intent_labels, multi_label=True)

    detected_intents = []
    confidence_scores = []

    for label, score in zip(result['labels'], result['scores']):
        if score >= threshold:
            detected_intents.append(label)
            confidence_scores.append(round(score, 3))

    return {
        'intents': detected_intents,
        'scores': confidence_scores,
        'top_intent': result['labels'][0],
        'top_score': round(result['scores'][0], 3)
    }


def classify_with_gemini(gemini_model, text, intent_labels):
    """Classify using Gemini API as fallback"""

    prompt = f"""You are an appointment scheduling intent classifier. Analyze the customer message and classify their intent.

Intent Categories:
1. schedule_new_appointment - Customer wants to book a new appointment
2. confirm_existing_appointment - Customer has already scheduled and is confirming
3. inquire_appointment_duration - Customer asking how long the service takes
4. propose_time_slot - Customer suggesting specific dates/times
5. check_availability - Customer asking if certain times are available
6. general_inquiry - Other appointment-related questions

Customer Message: "{text}"

Respond ONLY with a JSON object in this exact format (no markdown, no explanation):
{{"top_intent": "intent_name", "confidence": 0.XX, "all_intents": ["intent1", "intent2"]}}

Choose the most appropriate intent(s) from the list above."""

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        # Remove markdown code blocks if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()

        # Parse JSON response
        result = json.loads(response_text)

        return {
            'top_intent': result.get('top_intent', 'general_inquiry'),
            'top_score': result.get('confidence', 0.5),
            'intents': result.get('all_intents', [result.get('top_intent')]),
            'scores': [result.get('confidence', 0.5)]
        }

    except Exception as e:
        print(f"  ⚠️  Gemini API error: {e}")
        # Return fallback
        return {
            'top_intent': 'general_inquiry',
            'top_score': 0.3,
            'intents': ['general_inquiry'],
            'scores': [0.3]
        }


def hybrid_classify(text, local_classifier, gemini_model, intent_labels, confidence_threshold=0.7):
    """
    Hybrid classification: Local first, Gemini fallback
    """

    # Step 1: Try local model
    local_result = classify_with_local_model(local_classifier, text, intent_labels)

    # Step 2: Check if confident enough
    if local_result['top_score'] >= confidence_threshold:
        # Use local result
        return {
            **local_result,
            'source': 'local_only',
            'local_intent': local_result['top_intent'],
            'local_confidence': local_result['top_score'],
            'api_used': False
        }

    # Step 3: Low confidence - use Gemini API
    if gemini_model is None:
        # No API available, use local anyway
        return {
            **local_result,
            'source': 'local_fallback',
            'local_intent': local_result['top_intent'],
            'local_confidence': local_result['top_score'],
            'api_used': False
        }

    print(f"    → Low confidence ({local_result['top_score']:.2f}), using Gemini API...")

    # Add small delay to avoid rate limits
    time.sleep(0.5)

    gemini_result = classify_with_gemini(gemini_model, text, intent_labels)

    return {
        'top_intent': gemini_result['top_intent'],
        'top_score': gemini_result['top_score'],
        'intents': gemini_result['intents'],
        'scores': gemini_result['scores'],
        'source': 'api_verified',
        'local_intent': local_result['top_intent'],
        'local_confidence': local_result['top_score'],
        'api_used': True
    }


def process_appointments(
        csv_path='chatX.csv',
        screenshot_csv_path='screenshot_conversations.csv',
        output_path='all_conversations_with_intents.csv',
        confidence_threshold=0.7,
        use_hybrid=True
):
    """
    Main function to process all appointment sentences and classify intents

    Args:
        csv_path: Path to chatX.csv
        screenshot_csv_path: Path to screenshot_conversations.csv
        output_path: Path to save results CSV
        confidence_threshold: Threshold for hybrid mode (0.7 recommended)
        use_hybrid: If True, use Gemini API for low confidence cases
    """

    # Define intent categories for appointment conversations
    intent_labels = [
        "schedule_new_appointment",
        "confirm_existing_appointment",
        "inquire_appointment_duration",
        "propose_time_slot",
        "check_availability",
        "general_inquiry"
    ]

    mode = "HYBRID (Local + Gemini API)" if use_hybrid else "LOCAL ONLY"

    print("=" * 70)
    print(f"APPOINTMENT INTENT CLASSIFICATION - {mode}")
    print("=" * 70)
    print(f"\nIntent Categories:")
    for i, label in enumerate(intent_labels, 1):
        print(f"  {i}. {label}")
    print(f"\nConfidence Threshold: {confidence_threshold * 100}%")
    if use_hybrid:
        print(f"  (Gemini API used when local confidence < {confidence_threshold * 100}%)")
    print("=" * 70 + "\n")

    # Load data
    df = load_data(csv_path, screenshot_csv_path)

    # Initialize local classifier
    local_classifier = initialize_classifier()

    # Initialize Gemini if hybrid mode
    gemini_model = initialize_gemini() if use_hybrid else None

    # Process each sentence
    results = []
    api_call_count = 0
    local_only_count = 0

    print("Processing conversations...")
    print("-" * 70)

    for idx, row in df.iterrows():
        text = row['Text']
        source = row['Source']
        index = row['Index']

        print(f"\n[{idx + 1}/{len(df)}] {source} #{index}")
        print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        # Classify
        if use_hybrid:
            classification = hybrid_classify(
                text,
                local_classifier,
                gemini_model,
                intent_labels,
                confidence_threshold
            )
        else:
            # Local only mode
            classification = classify_with_local_model(local_classifier, text, intent_labels)
            classification['source'] = 'local_only'
            classification['api_used'] = False
            classification['local_intent'] = classification['top_intent']
            classification['local_confidence'] = classification['top_score']

        # Track API usage
        if classification.get('api_used', False):
            api_call_count += 1
        else:
            local_only_count += 1

        # Display result
        print(f"  ✓ Intent: {classification['top_intent']} ({classification['top_score']:.3f})")
        print(f"    Source: {classification['source']}")
        if classification.get('api_used', False):
            print(f"    Local was: {classification['local_intent']} ({classification['local_confidence']:.3f})")

        # Store results
        results.append({
            'Data_Source': source,
            'Original_Index': index,
            'Conversation_Text': text,
            'Final_Intent': classification['top_intent'],
            'Final_Confidence': classification['top_score'],
            'All_Intents': ', '.join(classification['intents']),
            'Classification_Source': classification['source'],
            'Local_Intent': classification.get('local_intent', ''),
            'Local_Confidence': classification.get('local_confidence', ''),
            'API_Used': classification.get('api_used', False)
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 SUMMARY:")
    print(f"  Total conversations processed: {len(df)}")
    print(f"  ✓ Handled by local model only: {local_only_count} ({local_only_count / len(df) * 100:.1f}%)")
    if use_hybrid:
        print(f"  ✓ Required Gemini API verification: {api_call_count} ({api_call_count / len(df) * 100:.1f}%)")

        # Cost estimation
        total_chars = sum(len(row['Text']) for _, row in df.iterrows() if results[_]['API_Used'])
        estimated_cost = (total_chars / 1000) * 0.0005
        print(f"\n💰 COST ANALYSIS:")
        print(f"  API calls made: {api_call_count}")
        print(f"  Estimated cost: ${estimated_cost:.4f}")
        print(f"  Cost saved by local model: ${((len(df) - api_call_count) * 0.0005):.4f}")

    # Intent distribution
    print(f"\n📈 INTENT DISTRIBUTION:")
    intent_counts = results_df['Final_Intent'].value_counts()
    for intent, count in intent_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {intent}: {count} ({percentage:.1f}%)")

    print(f"\n✓ Results saved to: {output_path}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    # Configuration
    CHATX_CSV = "chatX.csv"
    SCREENSHOT_CSV = "screenshot_conversations.csv"
    OUTPUT_CSV = "all_conversations_with_intents.csv"
    CONFIDENCE_THRESHOLD = 0.7  # Adjust: 0.5-0.8 range
    USE_HYBRID = True  # Set to False to use local model only

    # Run classification
    results = process_appointments(
        csv_path=CHATX_CSV,
        screenshot_csv_path=SCREENSHOT_CSV,
        output_path=OUTPUT_CSV,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        use_hybrid=USE_HYBRID
    )

    print("\n🎯 Sample Results (first 5 rows):")
    print(results[['Data_Source', 'Original_Index', 'Final_Intent', 'Final_Confidence',
                   'Classification_Source']].head().to_string())