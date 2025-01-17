from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
sys.path.append('..')
from run import process_single_paper, BertEmbeddings, ConferenceRecommender, GeminiAnalyzer, download_pdf_from_drive
from src.utils.text import extract_text_from_pdf
from conference_recommender.src.data_loader import ConferenceDataLoader
from conference_recommender.src.vector_store import ConferenceVectorStore
import joblib
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
import tempfile
import logging
from multiprocessing import Process, Manager
import pandas as pd
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from flask_socketio import SocketIO, emit
import psutil
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def emit_progress(message, data=None):
    """Emit progress update to frontend."""
    if data is None:
        data = {}
    socketio.emit('progress_update', {'message': message, **data})
    logger.info(message)

def get_gdrive_service():
    """Set up and return Google Drive service with specific redirect URI."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Use specific redirect URI
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', 
                SCOPES,
                redirect_uri='http://localhost:8080/'
            )
            creds = flow.run_local_server(port=8080)
            
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'Train')  # For publishability data
CONFERENCE_DATA_PATH = os.path.join(ROOT_DIR, 'conference_recommender', 'conference_training')  # For conference data

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model storage
model_store = {}

def extract_folder_id(drive_link):
    """Extract folder ID from Google Drive link."""
    # Handle different Google Drive link formats
    patterns = [
        r'folders/([a-zA-Z0-9-_]+)',  # Regular folder link
        r'id=([a-zA-Z0-9-_]+)',       # Alternate format
        r'/d/([a-zA-Z0-9-_]+)'        # Short format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_link)
        if match:
            return match.group(1)
    
    return None

def process_drive_folder(drive_link, models):
    """Process all PDFs in a Google Drive folder."""
    try:
        # Extract folder ID from link
        folder_id = extract_folder_id(drive_link)
        if not folder_id:
            raise ValueError("Invalid Google Drive folder link")
            
        # Get Google Drive service
        emit_progress("Initializing Google Drive service...")
        service = get_gdrive_service()
        
        # List files in the folder
        emit_progress(f"Scanning Google Drive folder {folder_id} for PDF files...")
        query = f"'{folder_id}' in parents and mimeType='application/pdf'"
        results = service.files().list(
            q=query,
            fields="files(id, name)",
            pageSize=1000
        ).execute()
        files = results.get('files', [])
        
        if not files:
            raise ValueError("No PDF files found in the folder")
        
        total_files = len(files)
        emit_progress(f"Found {total_files} PDF files to process", {
            'total_files': total_files,
            'current_file': 0,
            'processed_files': 0
        })
        
        # Initialize metrics
        processed_files = []
        total_time = 0
        success_count = 0
        publishable_count = 0
        conference_distribution = {}
        quality_scores = {
            'writing': [], 'methodology': [], 'innovation': [],
            'technical': [], 'validation': [], 'coherence': []
        }
        
        # Create a temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            results = []
            for idx, file in enumerate(files, 1):
                try:
                    filename = file['name']
                    file_id = file['id']
                    filepath = os.path.join(temp_dir, filename)
                    
                    emit_progress(f"Downloading file {idx}/{total_files}: {filename}")
                    
                    # Download the file
                    request = service.files().get_media(fileId=file_id)
                    with open(filepath, 'wb') as f:
                        downloader = MediaIoBaseDownload(f, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                            if status:
                                emit_progress(f"Download progress: {int(status.progress() * 100)}%")
                    
                    emit_progress(f"Processing file {idx}/{total_files}: {filename}", {
                        'current_file': idx,
                        'filename': filename
                    })
                    
                    # Process the paper
                    start_time = time.time()
                    result = process_single_paper(
                        filepath=filepath,
                        bert_model=models['bert_model'],
                        classifier_pipeline=models['classifier_pipeline'],
                        feature_columns=models['feature_columns'],
                        recommender=models['recommender'],
                        gemini_analyzer=models['gemini_analyzer']
                    )
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    # Update metrics
                    processed_files.append(filename)
                    success_count += 1
                    if result['publishable']:
                        publishable_count += 1
                    
                    # Update conference distribution
                    conf = result['conference']
                    conference_distribution[conf] = conference_distribution.get(conf, 0) + 1
                    
                    # Update quality scores
                    for score_type in quality_scores:
                        if f'{score_type}_score' in result:
                            quality_scores[score_type].append(result[f'{score_type}_score'])
                    
                    # Log and emit the results
                    score = result['publishability_score'] * 100
                    emit_progress(f"Results for {filename}:", {
                        'filename': filename,
                        'score': score,
                        'publishable': result['publishable'],
                        'conference': result['conference'],
                        'justification': result['justification']
                    })
                    
                    results.append(result)
                    emit_progress(f"Completed processing {filename}", {
                        'processed_files': idx
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    emit_progress(error_msg, {'error': True})
                    logger.error(error_msg)
                finally:
                    # Clean up the downloaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
        
        # Store metrics in model_store
        model_store['processed_files'] = processed_files
        model_store['avg_processing_time'] = total_time / len(processed_files) if processed_files else 0
        model_store['success_rate'] = (success_count / total_files) * 100
        model_store['publishable_count'] = publishable_count
        model_store['conference_distribution'] = conference_distribution
        model_store['quality_scores'] = {
            k: sum(v) / len(v) if v else 0 
            for k, v in quality_scores.items()
        }
        
        emit_progress(f"Batch processing complete. Processed {len(results)}/{total_files} files successfully.", {
            'complete': True,
            'total_processed': len(results)
        })
        return results
        
    except Exception as e:
        error_msg = f"Error processing drive folder: {str(e)}"
        emit_progress(error_msg, {'error': True})
        logger.error(error_msg)
        raise

def load_models():
    """Load all required models and components."""
    try:
        # Initialize BERT model
        bert_model = BertEmbeddings()
        
        # Check if model files exist
        classifier_path = os.path.join(ROOT_DIR, 'publishability_classifier.joblib')
        conference_path = os.path.join(ROOT_DIR, 'conference_classifier.joblib')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier model not found at {classifier_path}")
        if not os.path.exists(conference_path):
            raise FileNotFoundError(f"Conference classifier not found at {conference_path}")
            
        # Load models
        classifier_pipeline = joblib.load(classifier_path)
        
        # Get feature names from the classifier to ensure exact order
        if hasattr(classifier_pipeline, 'feature_names_in_'):
            feature_columns = classifier_pipeline.feature_names_in_.tolist()
            logger.info(f"Using {len(feature_columns)} features from classifier")
        else:
            # If feature names not available in classifier, use default order
            quality_scores = [
                'argument_coherence_score',
                'innovation_level_score',
                'methodology_quality_score',
                'result_validation_score',
                'technical_depth_score',
                'writing_quality_score'
            ]
            bert_features = [f'bert_{i}' for i in range(768)]
            feature_columns = quality_scores + bert_features
            logger.warning("Using default feature order - this might cause issues if it doesn't match the classifier")
        
        # Initialize conference recommendation components
        if not os.path.exists(CONFERENCE_DATA_PATH):
            raise FileNotFoundError(f"Conference data directory not found at {CONFERENCE_DATA_PATH}")
            
        # Load conference papers
        logger.info("Loading conference papers...")
        data_loader = ConferenceDataLoader(base_path=CONFERENCE_DATA_PATH)
        vector_store = ConferenceVectorStore()
        
        # Load papers from each conference directory
        conference_papers = {}
        for conf in ['CVPR', 'NeurIPS', 'EMNLP', 'TMLR', 'KDD']:
            conf_dir = os.path.join(CONFERENCE_DATA_PATH, conf)
            if os.path.exists(conf_dir):
                papers = []
                for filename in os.listdir(conf_dir):
                    if filename.endswith('.pdf'):
                        filepath = os.path.join(conf_dir, filename)
                        try:
                            with open(filepath, 'rb') as f:
                                text = extract_text_from_pdf(filepath)
                                papers.append({
                                    'text': text,
                                    'filename': filename
                                })
                        except Exception as e:
                            logger.warning(f"Failed to load {filename}: {str(e)}")
                conference_papers[conf] = papers
                logger.info(f"Loaded {len(papers)} papers from {conf}")
        
        # Add papers to vector store
        vector_store.add_papers(conference_papers)
        logger.info("Conference papers loaded and vectorized")
        
        # Initialize recommender with the prepared components
        recommender = ConferenceRecommender(vector_store=vector_store, data_loader=data_loader)
        
        # Initialize Gemini analyzer
        gemini_analyzer = GeminiAnalyzer()
        
        return {
            'bert_model': bert_model,
            'classifier_pipeline': classifier_pipeline,
            'feature_columns': feature_columns,
            'recommender': recommender,
            'gemini_analyzer': gemini_analyzer
        }
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def get_models():
    """Get or initialize models."""
    global model_store
    if not model_store:
        logger.info("Loading models for the first time...")
        model_store = load_models()
        logger.info("Models loaded successfully")
    return model_store

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def app_page():
    try:
        models = get_models()
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return render_template('index.html', error="Model initialization failed. Please check server logs.")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        models = get_models()
    except Exception as e:
        return jsonify({'success': False, 'error': 'Models not properly initialized'})
        
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the paper using the initialized components
            results = process_single_paper(
                filepath=filepath,
                bert_model=models['bert_model'],
                classifier_pipeline=models['classifier_pipeline'],
                feature_columns=models['feature_columns'],
                recommender=models['recommender'],
                gemini_analyzer=models['gemini_analyzer']
            )
            return jsonify({'success': True, 'results': results})
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})
        finally:
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/analyze-drive', methods=['POST'])
def analyze_drive():
    try:
        models = get_models()
    except Exception as e:
        return jsonify({'success': False, 'error': 'Models not properly initialized'})
        
    drive_link = request.form.get('drive_link')
    if not drive_link:
        return jsonify({'success': False, 'error': 'No Google Drive link provided'})
    
    try:
        # Process all PDFs in the drive folder
        results = process_drive_folder(drive_link, models)
        if not results:
            return jsonify({'success': False, 'error': 'No results from folder processing'})
            
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logger.error(f"Error processing drive link: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/performance')
def performance():
    try:
        # Get latest batch metrics
        batch_metrics = {
            'total_papers': len(model_store.get('processed_files', [])),
            'avg_time': model_store.get('avg_processing_time', 0),
            'success_rate': model_store.get('success_rate', 0),
            'publishable_count': model_store.get('publishable_count', 0),
            'conference_distribution': model_store.get('conference_distribution', {}),
            'quality_scores': model_store.get('quality_scores', {}),
            'resource_usage': {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent
            }
        }
        return render_template('performance.html', metrics=batch_metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return render_template('performance.html', error="Failed to load metrics")

if __name__ == '__main__':
    # Load models at startup in the main process
    logger.info("Loading models at startup...")
    model_store = load_models()
    logger.info("Models loaded successfully")
    
    # Run the app with Socket.IO on all IPs (0.0.0.0)
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, use_reloader=False) 