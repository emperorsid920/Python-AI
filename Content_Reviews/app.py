from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import json
import logging
from datetime import datetime
import os
import threading
import time
from werkzeug.utils import secure_filename

# Import our custom modules
from database import DatabaseManager
from data import DataProcessor
from models import ContentModerationModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize our components
db_manager = DatabaseManager()
data_processor = DataProcessor()
moderation_model = ContentModerationModel()

# Global variables for tracking processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'total': 0,
    'current_file': None,
    'errors': []
}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}


@app.route('/')
def dashboard():
    """
    Main dashboard showing overview of content moderation system.
    """
    try:
        # Get analytics data from database
        analytics = db_manager.get_analytics_data()

        # Get recent flagged content
        flagged_content = db_manager.get_flagged_content(limit=10)

        return render_template('dashboard.html',
                               analytics=analytics,
                               flagged_content=flagged_content)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('dashboard.html', analytics={}, flagged_content=[])


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """
    Handle file upload and initial processing.
    """
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']

        # Check if file is selected and valid
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Process the file in background
                threading.Thread(target=process_uploaded_file, args=(filepath,)).start()

                flash('File uploaded successfully! Processing started in background.', 'success')
                return redirect(url_for('processing_status'))

            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                flash(f'Error uploading file: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a CSV file.', 'error')
            return redirect(request.url)

    return render_template('upload.html')


def process_uploaded_file(filepath):
    """
    Process uploaded CSV file in background.

    Args:
        filepath (str): Path to uploaded CSV file
    """
    global processing_status

    try:
        processing_status['is_processing'] = True
        processing_status['current_file'] = os.path.basename(filepath)
        processing_status['progress'] = 0
        processing_status['errors'] = []

        logger.info(f"Starting to process file: {filepath}")

        # Load and validate CSV data
        df = data_processor.load_csv_data(filepath)
        processing_status['total'] = len(df)

        # Extract features
        logger.info("Extracting features...")
        df_features = data_processor.extract_features(df)
        processing_status['progress'] = 20

        # Create sentiment labels for training data
        df_features = data_processor.create_sentiment_labels(df_features)
        processing_status['progress'] = 30

        # Detect potential spam
        df_features = data_processor.detect_potential_spam(df_features)
        processing_status['progress'] = 40

        # Store reviews in database
        logger.info("Storing reviews in database...")
        db_manager.bulk_insert_reviews(df_features)
        processing_status['progress'] = 50

        # Load ML models if not already loaded
        if not moderation_model.sentiment_model:
            logger.info("Loading ML models...")
            moderation_model.load_pretrained_models()

        processing_status['progress'] = 60

        # Process reviews with ML models
        logger.info("Running ML predictions...")
        reviews_to_process = db_manager.get_reviews_for_processing()

        for i, review in enumerate(reviews_to_process):
            try:
                # Get features for this review
                review_features = df_features[df_features['review_id'] == review['review_id']].iloc[0].to_dict()

                # Run ML predictions
                predictions = moderation_model.predict_all(review['review_text'], review_features)

                # Store predictions
                db_manager.insert_prediction(review['review_id'], predictions)

                # Check if content should be flagged for review
                if predictions['is_spam'] or predictions['is_toxic'] or predictions['sentiment_label'] == 'negative':
                    priority = 3 if predictions['is_toxic'] else 2 if predictions['is_spam'] else 1
                    reasons = []

                    if predictions['is_toxic']:
                        reasons.append(f"High toxicity score: {predictions['toxicity_score']:.2f}")
                    if predictions['is_spam']:
                        reasons.append(f"Spam probability: {predictions['spam_probability']:.2f}")
                    if predictions['sentiment_label'] == 'negative':
                        reasons.append(f"Negative sentiment: {predictions['sentiment_score']:.2f}")

                    db_manager.add_to_moderation_queue(
                        review['review_id'],
                        "; ".join(reasons),
                        priority
                    )

                # Update progress
                processing_status['progress'] = 60 + (i / len(reviews_to_process)) * 35

            except Exception as e:
                logger.error(f"Error processing review {review['review_id']}: {e}")
                processing_status['errors'].append(f"Error processing review {review['review_id']}: {str(e)}")

        processing_status['progress'] = 100
        logger.info(f"File processing completed: {filepath}")

    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        processing_status['errors'].append(f"Critical error: {str(e)}")

    finally:
        processing_status['is_processing'] = False


@app.route('/processing-status')
def processing_status_page():
    """Show processing status page."""
    return render_template('processing_status.html')


@app.route('/api/processing-status')
def get_processing_status():
    """API endpoint to get current processing status."""
    return jsonify(processing_status)


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_text():
    """
    Analyze individual text input.
    """
    if request.method == 'POST':
        try:
            text = request.form.get('text', '').strip()

            if not text:
                flash('Please enter some text to analyze', 'error')
                return redirect(request.url)

            # Load models if not already loaded
            if not moderation_model.sentiment_model:
                moderation_model.load_pretrained_models()

            # Extract features for the text
            temp_df = pd.DataFrame([{
                'review_id': 'temp_analysis',
                'review_text': text,
                'rating': None,
                'date': datetime.now(),
                'reviewer_name': 'Anonymous'
            }])

            features_df = data_processor.extract_features(temp_df)
            features = features_df.iloc[0].to_dict()

            # Run predictions
            predictions = moderation_model.predict_all(text, features)

            return render_template('analyze.html',
                                   text=text,
                                   predictions=predictions,
                                   features=features)

        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            flash(f'Error analyzing text: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('analyze.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    API endpoint for text analysis.
    """
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Load models if not already loaded
        if not moderation_model.sentiment_model:
            moderation_model.load_pretrained_models()

        # Extract features
        temp_df = pd.DataFrame([{
            'review_id': 'api_analysis',
            'review_text': text,
            'rating': None,
            'date': datetime.now(),
            'reviewer_name': 'API'
        }])

        features_df = data_processor.extract_features(temp_df)
        features = features_df.iloc[0].to_dict()

        # Run predictions
        predictions = moderation_model.predict_all(text, features)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'features': {
                'text_length': features['text_length'],
                'word_count': features['word_count'],
                'caps_ratio': features['caps_ratio'],
                'spam_keyword_count': features['spam_keyword_count']
            }
        })

    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/moderation-queue')
def moderation_queue():
    """
    Show content flagged for human review.
    """
    try:
        # Get flagged content from database
        flagged_items = db_manager.get_flagged_content(limit=50)

        return render_template('moderation_queue.html', flagged_items=flagged_items)

    except Exception as e:
        logger.error(f"Error loading moderation queue: {e}")
        flash(f'Error loading moderation queue: {str(e)}', 'error')
        return render_template('moderation_queue.html', flagged_items=[])


@app.route('/analytics')
def analytics():
    """
    Show detailed analytics and insights.
    """
    try:
        # Get comprehensive analytics
        analytics_data = db_manager.get_analytics_data()

        return render_template('analytics.html', analytics=analytics_data)

    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        flash(f'Error loading analytics: {str(e)}', 'error')
        return render_template('analytics.html', analytics={})


@app.route('/api/batch-analyze', methods=['POST'])
def api_batch_analyze():
    """
    API endpoint for batch text analysis.
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])

        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Invalid texts array provided'}), 400

        if len(texts) > 100:  # Limit batch size
            return jsonify({'error': 'Batch size too large (max 100)'}), 400

        # Load models if not already loaded
        if not moderation_model.sentiment_model:
            moderation_model.load_pretrained_models()

        # Process each text
        results = []
        for i, text in enumerate(texts):
            try:
                # Extract features
                temp_df = pd.DataFrame([{
                    'review_id': f'batch_{i}',
                    'review_text': text,
                    'rating': None,
                    'date': datetime.now(),
                    'reviewer_name': 'Batch_API'
                }])

                features_df = data_processor.extract_features(temp_df)
                features = features_df.iloc[0].to_dict()

                # Run predictions
                predictions = moderation_model.predict_all(text, features)

                results.append({
                    'text': text,
                    'predictions': predictions
                })

            except Exception as e:
                logger.error(f"Error processing batch item {i}: {e}")
                results.append({
                    'text': text,
                    'error': str(e)
                })

        return jsonify({
            'success': True,
            'results': results,
            'processed_count': len(results)
        })

    except Exception as e:
        logger.error(f"Batch API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring.
    """
    try:
        # Check database connection
        db_manager.get_connection().close()

        # Check if models are loaded
        models_loaded = moderation_model.sentiment_model is not None

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'models_loaded': models_loaded,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html',
                           error_code=404,
                           error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html',
                           error_code=500,
                           error_message="Internal server error"), 500


if __name__ == '__main__':
    # Initialize models on startup
    logger.info("Initializing application...")

    try:
        # Load pre-trained models
        moderation_model.load_pretrained_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.warning("Application will start without pre-loaded models")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)