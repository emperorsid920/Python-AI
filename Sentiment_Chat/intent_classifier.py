"""
Intent Classification for Appointment Conversations
Uses DeBERTa v3 zero-shot classification model to detect multiple intents in appointment-related text
"""

import pandas as pd
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')


def load_data(csv_path):
    """Load the CSV file with appointment sentences"""
    df = pd.read_csv(csv_path)
    # Remove rows where Sentences column is empty/NaN
    df = df[df['Sentences'].notna()]
    return df


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


def classify_intents(classifier, text, candidate_labels, threshold=0.5):
    """
    Classify text with multiple possible intents

    Args:
        classifier: The zero-shot classification pipeline
        text: Input text to classify
        candidate_labels: List of possible intent labels
        threshold: Minimum confidence score to include an intent (0.5 = 50%)

    Returns:
        Dictionary with detected intents and their confidence scores
    """
    result = classifier(
        text,
        candidate_labels,
        multi_label=True  # Allow multiple intents per sentence
    )

    # Filter intents above threshold
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


def process_appointments(csv_path, output_path='chatX_with_intents.csv', confidence_threshold=0.5):
    """
    Main function to process all appointment sentences and classify intents

    Args:
        csv_path: Path to input CSV file
        output_path: Path to save results CSV
        confidence_threshold: Minimum confidence to include an intent (0-1)
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

    print("=" * 70)
    print("APPOINTMENT INTENT CLASSIFICATION")
    print("=" * 70)
    print(f"\nIntent Categories:")
    for i, label in enumerate(intent_labels, 1):
        print(f"  {i}. {label}")
    print(f"\nConfidence Threshold: {confidence_threshold * 100}%")
    print("=" * 70 + "\n")

    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} sentences from CSV\n")

    # Initialize classifier
    classifier = initialize_classifier()

    # Process each sentence
    results = []

    print("Processing sentences...")
    print("-" * 70)

    for idx, row in df.iterrows():
        sentence = row['Sentences']

        # Classify
        classification = classify_intents(
            classifier,
            sentence,
            intent_labels,
            threshold=confidence_threshold
        )

        # Display results
        print(f"\n[{idx + 1}] {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
        print(f"    Top Intent: {classification['top_intent']} ({classification['top_score']})")
        if len(classification['intents']) > 1:
            print(f"    All Intents: {', '.join(classification['intents'])}")

        # Store results
        results.append({
            'Index': row.iloc[0] if pd.notna(row.iloc[0]) else idx + 1,
            'Sentence': sentence,
            'Top_Intent': classification['top_intent'],
            'Top_Confidence': classification['top_score'],
            'All_Intents': ', '.join(classification['intents']),
            'All_Confidence_Scores': ', '.join(map(str, classification['scores']))
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print(f"✓ Processing complete!")
    print(f"✓ Results saved to: {output_path}")
    print("=" * 70)

    # Print summary statistics
    print("\n📊 INTENT DISTRIBUTION:")
    print("-" * 70)
    intent_counts = {}
    for intents_str in results_df['All_Intents']:
        for intent in intents_str.split(', '):
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results_df)) * 100
        print(f"  {intent}: {count} ({percentage:.1f}%)")

    return results_df


if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "chatX.csv"  # Change this to your CSV path if different
    OUTPUT_CSV = "chatX_with_intents.csv"
    CONFIDENCE_THRESHOLD = 0.5  # Adjust this (0.0 to 1.0) to be more/less strict

    # Run classification
    results = process_appointments(
        csv_path=INPUT_CSV,
        output_path=OUTPUT_CSV,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    print("\n🎯 Sample Results (first 3 rows):")
    print(results.head(3).to_string())