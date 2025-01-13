import os
import json
import logging
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from sklearn.pipeline import Pipeline

from src.utils.text import extract_text_from_pdf, segment_text
from src.embeddings.bert_embeddings import BertEmbeddings
from src.analysis.gemini_analysis import (
    get_gemini_analysis,
    combine_segment_analyses,
    get_publishability_decision,
    extract_features_from_analysis
)
from src.models.classifier import train_classifier, evaluate_classifier

# Initialize BERT embeddings model globally
bert_model = BertEmbeddings()

def process_papers(base_path: str) -> List[Dict]:
    """Process all papers in the dataset with segmentation and caching."""
    results = []
    
    # Process all papers
    for category in ['Publishable', 'Non-Publishable']:
        category_path = os.path.join(base_path, 'Train', category)
        files = [f for f in os.listdir(category_path) if f.endswith('.pdf')]
        logging.info(f"\nProcessing {len(files)} papers in {category} category")
        
        for filename in tqdm(files, desc=f"Processing {category} papers"):
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing paper: {filename}")
            paper_path = os.path.join(category_path, filename)
            
            # Extract text
            logging.info("Extracting text from PDF...")
            text = extract_text_from_pdf(paper_path)
            if not text:
                logging.error(f"Failed to extract text from {filename}")
                continue
            logging.info(f"Extracted {len(text)} characters of text")
            
            # Get BERT embedding
            logging.info("Generating BERT embedding...")
            bert_embedding = bert_model.get_embedding(text[:512])
            logging.info("BERT embedding generated successfully")
            
            # Segment text
            logging.info("Segmenting text...")
            segments = segment_text(text)
            logging.info(f"Text split into {len(segments)} segments")
            
            # Process each segment
            segment_analyses = []
            for i, segment in enumerate(segments, 1):
                logging.info(f"Analyzing segment {i}/{len(segments)} ({len(segment)} chars)")
                analysis = get_gemini_analysis(segment)
                if analysis:
                    segment_analyses.append(analysis)
                    logging.info(f"Segment {i} analysis completed")
                else:
                    logging.warning(f"Failed to analyze segment {i}")
            
            if not segment_analyses:
                logging.error(f"No successful analyses for {filename}")
                continue
            
            # Combine analyses
            logging.info("Combining segment analyses...")
            combined_analysis = combine_segment_analyses(segment_analyses)
            if not combined_analysis:
                logging.error("Failed to combine analyses")
                continue
            logging.info("Successfully combined analyses")
            
            # Get final decision
            logging.info("Getting publishability decision...")
            decision = get_publishability_decision(bert_embedding, combined_analysis)
            if not decision:
                logging.error("Failed to get decision")
                continue
            
            try:
                decision_data = json.loads(decision)
                logging.info(f"Decision: {decision_data.get('decision', 'UNKNOWN')} "
                           f"(Confidence: {decision_data.get('confidence_score', 'N/A')})")
            except json.JSONDecodeError:
                logging.warning("Could not parse decision JSON")
            
            # Store results
            results.append({
                'filename': filename,
                'actual_category': category,
                'segment_analyses': segment_analyses,
                'combined_analysis': combined_analysis,
                'decision': decision
            })
            
            # Save results after each paper
            logging.info("Saving results...")
            with open('paper_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Completed processing {filename}\n")
    
    return results

def prepare_training_data(results: List[Dict]) -> tuple:
    """Prepare training data using only BERT embeddings."""
    features_list = []
    labels = []
    
    logging.info("Extracting BERT embeddings for training...")
    for result in tqdm(results, desc="Processing training papers"):
        try:
            # Get BERT embedding from cache
            text = extract_text_from_pdf(
                os.path.join('data', 'Train', result['actual_category'], result['filename'])
            )
            if text is None:
                logging.warning(f"No text found for {result['filename']}")
                continue
                
            bert_embedding = bert_model.get_embedding(text[:512])
            if bert_embedding is None:
                logging.warning(f"No BERT embedding generated for {result['filename']}")
                continue
            
            # Use only BERT embeddings as features
            features = {f'bert_{i}': v for i, v in enumerate(bert_embedding.flatten())}
            
            features_list.append(features)
            labels.append(1 if result['actual_category'] == 'Publishable' else 0)
            
            logging.info(f"Processed {result['filename']} - Label: {'Publishable' if labels[-1] == 1 else 'Non-Publishable'}")
            
        except Exception as e:
            logging.error(f"Error preparing data for {result['filename']}: {str(e)}")
            continue
    
    if not features_list:
        raise ValueError("No valid features extracted from results")
    
    # Convert to DataFrame
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    logging.info(f"\nFeature matrix shape: {X.shape}")
    logging.info(f"Number of features: {X.shape[1]}")
    logging.info(f"Number of samples: {X.shape[0]}")
    logging.info(f"Class distribution:")
    logging.info(f"- Publishable: {sum(y == 1)}")
    logging.info(f"- Non-Publishable: {sum(y == 0)}")
    
    return X, y

def classify_test_papers(pipeline: Pipeline, feature_columns: List[str], base_path: str = 'data/Test', limit: int = 20, threshold: float = 0.7) -> None:
    """Classify papers using only BERT embeddings with custom probability threshold."""
    test_results = []
    
    # Get sorted list of PDF files
    pdf_files = sorted([f for f in os.listdir(base_path) if f.endswith('.pdf')])
    
    # Apply limit if specified
    if limit is not None:
        pdf_files = pdf_files[:limit]
        logging.info(f"\nClassifying first {len(pdf_files)} papers from test directory")
    else:
        logging.info(f"\nClassifying all {len(pdf_files)} papers from test directory")
    
    logging.info(f"Using probability threshold: {threshold}")
    
    # Process PDF files
    for filename in tqdm(pdf_files, desc="Classifying test papers"):
        try:
            # Get cached data
            paper_path = os.path.join(base_path, filename)
            
            # Get BERT embedding from cache
            text = extract_text_from_pdf(paper_path)
            if text is None:
                logging.error(f"Failed to extract text from {filename}")
                continue
                
            # Generate embedding for test paper
            bert_embedding = bert_model.get_embedding(text[:512])
            
            # Create features dictionary
            features = {f'bert_{i}': v for i, v in enumerate(bert_embedding.flatten())}
            
            # Create DataFrame with same columns as training data
            X_test = pd.DataFrame([{col: features.get(col, 0) for col in feature_columns}])
            
            # Get prediction probabilities
            probability = pipeline.predict_proba(X_test)[0]
            
            # Apply threshold to probability
            prediction = 1 if probability[1] >= threshold else 0
            
            result = {
                'Filename': filename,
                'Prediction': 'Publishable' if prediction == 1 else 'Non-Publishable',
                'Confidence': f"{float(probability.max()):.3f}",
                'Prob_Publishable': f"{float(probability[1]):.3f}",
                'Prob_NonPublishable': f"{float(probability[0]):.3f}"
            }
            
            test_results.append(result)
            
        except Exception as e:
            logging.error(f"Error classifying test file {filename}: {str(e)}")
            continue
    
    # Create and display results table
    results_df = pd.DataFrame(test_results)
    print("\nTest Predictions:")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100)
    
    # Save test results
    with open('test_predictions.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Print summary statistics
    publishable = sum(1 for r in test_results if r['Prediction'] == 'Publishable')
    non_publishable = len(test_results) - publishable
    print(f"\nSummary Statistics (threshold = {threshold}):")
    print(f"Total papers processed: {len(test_results)}")
    print(f"Publishable papers: {publishable} ({publishable/len(test_results)*100:.1f}%)")
    print(f"Non-publishable papers: {non_publishable} ({non_publishable/len(test_results)*100:.1f}%)")
    print(f"Average confidence: {sum(float(r['Confidence']) for r in test_results) / len(test_results):.3f}")
    
    # Print distribution of probabilities
    prob_publishable = [float(r['Prob_Publishable']) for r in test_results]
    print("\nProbability Distribution:")
    print(f"Mean probability of being publishable: {np.mean(prob_publishable):.3f}")
    print(f"Median probability of being publishable: {np.median(prob_publishable):.3f}")
    print(f"Std dev of probabilities: {np.std(prob_publishable):.3f}")
    
    # Print feature importance if using RandomForest
    if hasattr(pipeline['classifier'], 'feature_importances_'):
        importances = pipeline['classifier'].feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important BERT Features:")
        print(feature_imp.head(10).to_string(index=False))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Paper Classification System')
    parser.add_argument('--skip-training', action='store_true', 
                      help='Skip training and load existing model')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Probability threshold for publishable classification (default: 0.7)')
    args = parser.parse_args()
    
    logging.info("Starting paper classification using BERT embeddings...")
    logging.info(f"Using probability threshold: {args.threshold}")
    
    # Load cached results for evaluation regardless of training mode
    if not os.path.exists('paper_analysis_results.json'):
        logging.error("No cached analysis results found. Please run the analysis pipeline first.")
        return
    
    with open('paper_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    # Prepare data for evaluation
    logging.info("\nPreparing data for evaluation...")
    X, y = prepare_training_data(results)
    
    if args.skip_training:
        # Load existing model
        if not os.path.exists('publishability_classifier_bert.joblib'):
            logging.error("No trained model found. Please run without --skip-training first.")
            return
        logging.info("Loading existing model...")
        pipeline = joblib.load('publishability_classifier_bert.joblib')
        
        # Load feature columns from saved model
        feature_columns = pipeline.feature_names_in_.tolist()
        
    else:
        logging.info("\nTraining classifier with balanced data...")
        pipeline = train_classifier(X, y)
        feature_columns = X.columns.tolist()
        
        # Save the trained model
        logging.info("\nSaving trained model...")
        joblib.dump(pipeline, 'publishability_classifier_bert.joblib')
    
    # Always show evaluation metrics
    logging.info("\nEvaluating classifier...")
    evaluate_classifier(pipeline, X, y)
    
    logging.info("\nClassifying all test papers...")
    classify_test_papers(pipeline, feature_columns, limit=None, threshold=args.threshold)
    
    logging.info("\nProcess completed!")

if __name__ == "__main__":
    main()
