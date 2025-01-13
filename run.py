import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from PyPDF2 import PdfReader
import argparse
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import necessary components from both systems
from src.utils.text import extract_text_from_pdf, segment_text
from src.embeddings.bert_embeddings import BertEmbeddings
from src.analysis.gemini_analysis import get_gemini_analysis, combine_segment_analyses
from conference_recommender.src.data_loader import ConferenceDataLoader
from conference_recommender.src.vector_store import ConferenceVectorStore
from conference_recommender.src.recommender import ConferenceRecommender
from conference_recommender.src.gemini_analyzer import GeminiAnalyzer

# Set up logging
os.makedirs('outputs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'outputs/run_output_{timestamp}.txt'

# Create handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Also capture print statements
import sys
class PrintCapture:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
    
    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, 'a') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()

sys.stdout = PrintCapture(log_file)

def get_gdrive_service():
    """Set up and return Google Drive service."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = None
    
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def download_pdf_from_drive(service, folder_id: str) -> list:
    """Download PDFs from Google Drive folder."""
    results = []
    page_token = None
    
    # Create temporary directory for downloaded files
    temp_dir = Path('temp_pdfs')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        while True:
            # List files in the folder
            response = service.files().list(
                q=f"'{folder_id}' in parents and mimeType='application/pdf'",
                spaces='drive',
                fields='nextPageToken, files(id, name)',
                pageToken=page_token
            ).execute()
            
            for file in response.get('files', []):
                file_id = file['id']
                file_name = file['name']
                file_path = temp_dir / file_name
                
                # Download file
                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                # Save to temporary file
                fh.seek(0)
                with open(file_path, 'wb') as f:
                    f.write(fh.read())
                
                results.append(str(file_path))
            
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break
                
    except Exception as e:
        logging.error(f"Error downloading files from Google Drive: {str(e)}")
    
    return results

def convert_numpy_types(obj):
    """Convert numpy types to native Python types in a nested structure."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    return obj

def process_single_paper(filepath: str, bert_model: BertEmbeddings, 
                        classifier_pipeline: object, feature_columns: list,
                        recommender: ConferenceRecommender = None,
                        gemini_analyzer: GeminiAnalyzer = None,
                        threshold: float = 0.7) -> dict:
    """Process a single paper through both classifiers."""
    start_time = time.time()
    
    result = {
        'pdfname': os.path.basename(filepath),
        'publishable': 0,
        'conference': 'NA',
        'justification': 'NA',
        'publishability_score': 0.0,
        'processing_time': 0.0
    }
    
    try:
        # Extract text
        text = extract_text_from_pdf(filepath)
        if not text:
            logging.error("Failed to extract text from PDF")
            return result
            
        # Get BERT embedding for publishability classification
        bert_embedding = bert_model.get_embedding(text[:512])
        if bert_embedding is None:
            logging.error("Failed to generate BERT embedding")
            return result
            
        # Create features dictionary and convert numpy types to native Python types
        features = {f'bert_{i}': float(v) for i, v in enumerate(bert_embedding.flatten())}
        
        # Create DataFrame with same columns as training data
        X_test = pd.DataFrame([{col: features.get(col, 0) for col in feature_columns}])
        
        # Get prediction probabilities
        probability = classifier_pipeline.predict_proba(X_test)[0]
        prediction = 1 if probability[1] >= threshold else 0
        
        # Update result with publishability
        result['publishable'] = int(prediction)
        result['publishability_score'] = float(probability[1])
        
        # If paper is classified as publishable and recommender is provided
        if prediction == 1 and recommender is not None:
            # Get vector-based analysis
            vector_analysis = recommender.analyze_paper(text)
            
            # Convert numpy types in vector analysis to native Python types
            if vector_analysis:
                vector_analysis = convert_numpy_types(vector_analysis)
            
            # Get Gemini analysis if available
            gemini_analysis = None
            if gemini_analyzer is not None:
                try:
                    gemini_analysis = gemini_analyzer.analyze_paper(text)
                except Exception as e:
                    logging.error(f"Gemini analysis failed: {str(e)}")
            
            # Combine recommendations if Gemini analysis is available
            if gemini_analysis is not None and gemini_analyzer is not None:
                try:
                    # Convert both analyses to native Python types before combining
                    gemini_analysis = convert_numpy_types(gemini_analysis)
                    final_analysis = gemini_analyzer.combine_recommendations(
                        gemini_result=gemini_analysis,
                        model_result=vector_analysis
                    )
                    # Ensure we use Gemini's justification if available
                    if isinstance(gemini_analysis, dict) and 'justification' in gemini_analysis:
                        final_analysis['justification'] = gemini_analysis['justification']
                except Exception as e:
                    logging.error(f"Failed to combine analyses: {str(e)}")
                    final_analysis = gemini_analysis if gemini_analysis else vector_analysis
            else:
                final_analysis = vector_analysis
            
            if final_analysis:
                result['conference'] = str(final_analysis.get('recommended_conference', 'NA'))
                result['justification'] = str(final_analysis.get('justification', 'NA'))
        
    except Exception as e:
        logging.error(f"Error processing {filepath}: {str(e)}")
    
    result['processing_time'] = float(time.time() - start_time)
    
    # Print just the essential info in CSV format
    print(f"{result['pdfname']}, {result['publishable']}, {result['conference']}, {result['publishability_score']:.3f}, {result['processing_time']:.2f}")
    
    return result

def print_performance_metrics(results_df: pd.DataFrame):
    """Print detailed performance metrics."""
    print("\nPerformance Metrics:")
    print("=" * 100)
    
    # Processing time statistics
    proc_times = results_df['processing_time']
    print("\nProcessing Time Statistics (seconds):")
    print(f"Average: {proc_times.mean():.2f}")
    print(f"Median: {proc_times.median():.2f}")
    print(f"Min: {proc_times.min():.2f}")
    print(f"Max: {proc_times.max():.2f}")
    
    # Publishability score distribution
    pub_scores = results_df['publishability_score']
    print("\nPublishability Score Distribution:")
    print(f"Mean: {pub_scores.mean():.3f}")
    print(f"Median: {pub_scores.median():.3f}")
    print(f"Std Dev: {pub_scores.std():.3f}")
    
    # Conference recommendation statistics
    has_conf = results_df['conference'] != 'NA'
    print("\nConference Recommendation Statistics:")
    print(f"Papers with recommendations: {sum(has_conf)} ({sum(has_conf)/len(results_df)*100:.1f}%)")
    if sum(has_conf) > 0:
        print("\nTop Recommended Conferences:")
        conf_counts = results_df[has_conf]['conference'].value_counts()
        print(conf_counts.head().to_string())

def main():
    parser = argparse.ArgumentParser(description='Paper Analysis Pipeline')
    parser.add_argument('--gdrive-folder', type=str, help='Google Drive folder ID for input PDFs')
    parser.add_argument('--input-dir', type=str, default='data/Test',
                      help='Local input directory for PDFs (default: data/Test)')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Threshold for publishability classification (default: 0.7)')
    args = parser.parse_args()
    
    # Initialize BERT model first
    logging.info("Initializing BERT model...")
    bert_model = BertEmbeddings()
    
    # Load publishability classifier
    if not os.path.exists('publishability_classifier_bert.joblib'):
        logging.error("No trained publishability classifier found!")
        return
        
    logging.info("Loading publishability classifier...")
    pipeline = joblib.load('publishability_classifier_bert.joblib')
    
    # Show model metrics first
    print("\nEvaluating publishability classifier performance...")
    from src.models.classifier import evaluate_classifier
    
    # Load a sample of training data for evaluation
    train_data_path = 'data/Train'
    X_train = []
    y_train = []
    
    for category in ['Publishable', 'Non-Publishable']:
        category_path = os.path.join(train_data_path, category)
        if not os.path.exists(category_path):
            logging.error(f"Training data directory not found: {category_path}")
            return
            
        files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
        for file in files:
            filepath = os.path.join(category_path, file)
            text = extract_text_from_pdf(filepath)
            if text:
                embedding = bert_model.get_embedding(text[:512])
                if embedding is not None:
                    X_train.append(embedding.flatten())
                    y_train.append(1 if category == 'Publishable' else 0)
    
    if not X_train:
        logging.error("No training data could be loaded for evaluation!")
        return
        
    X_train = pd.DataFrame(X_train)
    X_train.columns = [f'bert_{i}' for i in range(X_train.shape[1])]
    y_train = np.array(y_train)
    
    # Evaluate using cross-validation
    evaluate_classifier(pipeline, X_train, y_train)
    
    feature_columns = pipeline.feature_names_in_.tolist()
    
    # Initialize conference recommender components
    logging.info("\nInitializing conference recommender...")
    data_loader = ConferenceDataLoader('conference_recommender/conference_training')
    vector_store = ConferenceVectorStore()
    
    # Load reference papers
    conference_papers = data_loader.load_papers()
    vector_store.add_papers(conference_papers)
    recommender = ConferenceRecommender(vector_store, data_loader)
    
    # Initialize Gemini analyzer
    gemini_analyzer = GeminiAnalyzer()
    
    # Process papers from either Google Drive or local directory
    results = []
    
    if args.gdrive_folder:
        logging.info(f"Setting up Google Drive connection for folder: {args.gdrive_folder}")
        service = get_gdrive_service()
        pdf_files = download_pdf_from_drive(service, args.gdrive_folder)
        
        for filepath in tqdm(pdf_files, desc="Processing papers from Google Drive"):
            result = process_single_paper(
                filepath=filepath,
                bert_model=bert_model,
                classifier_pipeline=pipeline,
                feature_columns=feature_columns,
                recommender=recommender,
                gemini_analyzer=gemini_analyzer,
                threshold=args.threshold
            )
            results.append(result)
            
        # Cleanup temporary files
        for filepath in pdf_files:
            try:
                os.remove(filepath)
            except:
                pass
        try:
            os.rmdir('temp_pdfs')
        except:
            pass
    else:
        input_dir = args.input_dir
        if not os.path.exists(input_dir):
            logging.error(f"Input directory {input_dir} not found!")
            return
            
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        logging.info(f"Found {len(pdf_files)} PDF files to process")
        
        for filename in tqdm(pdf_files, desc="Processing papers"):
            filepath = os.path.join(input_dir, filename)
            result = process_single_paper(
                filepath=filepath,
                bert_model=bert_model,
                classifier_pipeline=pipeline,
                feature_columns=feature_columns,
                recommender=recommender,
                gemini_analyzer=gemini_analyzer,
                threshold=args.threshold
            )
            results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Sort DataFrame alphabetically by pdfname
    df = df.sort_values('pdfname', ascending=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'combined_analysis_results_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    # Display results
    print("\nAnalysis Results:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # Print summary statistics
    print(f"\nTotal papers processed: {len(df)}")
    print(f"Publishable papers: {sum(df['publishable'] == 1)}")
    print(f"Non-publishable papers: {sum(df['publishable'] == 0)}")
    print(f"Papers with conference recommendations: {df['conference'].ne('NA').sum()}")
    
    # Print detailed performance metrics
    print_performance_metrics(df)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 