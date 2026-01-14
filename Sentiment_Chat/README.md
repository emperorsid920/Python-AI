# Appointment Intent Classification

A zero-shot classification system for identifying intents in appointment-related conversations using the DeBERTa v3 model from Hugging Face.

## Overview

This project uses state-of-the-art NLP to automatically classify appointment-related text into multiple intent categories without requiring any labeled training data. It's particularly useful for analyzing customer service conversations, scheduling inquiries, and appointment management workflows.

## Features

- **Zero-shot classification**: No training data required - just define your intent categories
- **Multi-label detection**: Identifies multiple intents in a single sentence
- **High accuracy**: Uses the robust DeBERTa v3 model trained on 33 different datasets
- **Confidence scoring**: Get probability scores for each detected intent
- **Easy to customize**: Adjust intent categories and confidence thresholds
- **CSV output**: Results saved in a clean, structured format

## Intent Categories

The system detects the following intents:

1. **schedule_new_appointment** - Customer wants to book a new appointment
2. **confirm_existing_appointment** - Customer has already scheduled and is confirming
3. **inquire_appointment_duration** - Customer asking how long the appointment will take
4. **propose_time_slot** - Customer suggesting specific dates/times
5. **check_availability** - Customer checking if certain times are available
6. **general_inquiry** - Other appointment-related questions

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required dependencies:

```bash
pip install transformers pandas torch
```

Or if using pip3:

```bash
pip3 install transformers pandas torch
```

**Note**: The first run will download the DeBERTa model (~1.5GB), which may take a few minutes depending on your internet connection.

## Usage

### Basic Usage

1. Place your CSV file with appointment sentences in the project directory (default: `chatX.csv`)

2. Run the classifier:

```bash
python intent_classifier.py
```

3. Results will be saved to `chatX_with_intents.csv`

### CSV Format

Your input CSV should have a column named `Sentences` containing the text to classify:

```csv
,Sentences,,,
1,"Can I schedule for 31st? How long is appointment? 11am?",,,
2,"I made an appointment for next Friday at 12. Thank you",,,
```

### Output Format

The output CSV includes:

- `Index`: Original row index
- `Sentence`: Original text
- `Top_Intent`: Most confident intent detected
- `Top_Confidence`: Confidence score for top intent (0-1)
- `All_Intents`: Comma-separated list of all detected intents
- `All_Confidence_Scores`: Corresponding confidence scores

### Customization

Edit the configuration section at the bottom of `intent_classifier.py`:

```python
# Configuration
INPUT_CSV = "chatX.csv"  # Change to your CSV filename
OUTPUT_CSV = "chatX_with_intents.csv"  # Change output filename
CONFIDENCE_THRESHOLD = 0.5  # Adjust threshold (0.0 to 1.0)
```

**Confidence Threshold**:
- `0.3` = More permissive (detects more intents, may include false positives)
- `0.5` = Balanced (default)
- `0.7` = Stricter (only high-confidence intents)

### Adding Custom Intents

To add or modify intent categories, edit the `intent_labels` list in the `process_appointments` function:

```python
intent_labels = [
    "schedule_new_appointment",
    "cancel_appointment",  # Add new intents here
    "reschedule_appointment",
    # ... your custom intents
]
```

## Project Structure

```
Sentiment_Chat/
├── chatX.csv                    # Input data
├── intent_classifier.py         # Main classification script
├── chatX_with_intents.csv      # Output results (generated)
└── README.md                    # This file
```

## Model Information

**Model**: [MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33)

**Architecture**: DeBERTa v3 Base  
**Training**: 33 diverse datasets covering various domains  
**Task**: Zero-shot text classification  
**License**: MIT

This model was chosen for its:
- Superior accuracy on zero-shot tasks
- Robustness across different domains
- Active maintenance and community support

## Troubleshooting

### Import Error: `No module named 'transformers'`

Install the transformers library:
```bash
pip install transformers
```

### TensorFlow/Keras Compatibility Error

The script explicitly uses PyTorch to avoid TensorFlow conflicts. If you still encounter issues, ensure you have PyTorch installed:
```bash
pip install torch
```

### Memory Issues

If you run out of memory, the model is running on CPU by default. For large datasets, consider:
1. Processing in smaller batches
2. Using a machine with more RAM
3. Using GPU if available (change `device=-1` to `device=0` in the code)

### Slow Performance

First run downloads the model and is slower. Subsequent runs are much faster. If still slow:
- Reduce the number of intent categories
- Process smaller batches of data
- Use GPU acceleration if available

## Performance

**Processing Speed**: ~1-2 seconds per sentence on CPU  
**Accuracy**: High accuracy for appointment-related text (exact metrics depend on your use case)  
**Scalability**: Can process hundreds of sentences in a reasonable timeframe

## Future Enhancements

Potential improvements for this project:

- [ ] Batch processing for faster performance
- [ ] Web interface for non-technical users
- [ ] Integration with appointment management systems
- [ ] Fine-tuning on domain-specific data
- [ ] Real-time classification API
- [ ] Visualization dashboard for intent trends

## Contributing

Feel free to fork this project and submit pull requests for improvements!

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Model by Moritz Laurer: [Hugging Face Profile](https://huggingface.co/MoritzLaurer)
- Built with [Transformers](https://github.com/huggingface/transformers) by Hugging Face
- Powered by [PyTorch](https://pytorch.org/)

## Contact

For questions or issues, please open an issue in the repository.

---

**Made with ❤️ for better appointment management**
