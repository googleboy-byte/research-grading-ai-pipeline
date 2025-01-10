import pathway as pw
from data_processing import process_directory
from models import extract_features, ConferenceRecommender, PublishabilityClassifier
from utils import (
    PAPERS_PATH, PUBLISHABLE_PATH, NON_PUBLISHABLE_PATH,
    TRAIN_PUBLISHABLE_PATH, TRAIN_NON_PUBLISHABLE_PATH,
    get_api_keys
)
import json
from pathlib import Path
import logging
import numpy as np
import argparse
import pandas as pd
from termcolor import colored
from tabulate import tabulate
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from datetime import datetime
from models.rag import ResearchPaperRAG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 10  # Number of papers to process from each conference
MIN_SAMPLES_PER_CONFERENCE = 1  # Minimum required samples per conference

def get_sample_paths(directory_path: Path, size: int = SAMPLE_SIZE, is_test: bool = False) -> tuple[list[str], list[int]]:
    """Get sample paths from a directory, recursively searching subdirectories
    Returns a tuple of (paths, labels) where labels are 1 for publishable papers and 0 for non-publishable
    
    Args:
        directory_path: Base directory to search
        size: Number of samples to return
        is_test: Whether this is test data (affects labeling strategy)
    """
    try:
        if not directory_path.exists():
            logger.warning(f"Directory does not exist: {directory_path}")
            return [], []
            
        # Recursively search for PDFs in all subdirectories
        paths = list(directory_path.rglob("*.pdf"))
        if not paths:
            logger.warning(f"No PDF files found in {directory_path}")
            return [], []
            
        # Convert Path objects to strings
        str_paths = [str(p) for p in paths]
        
        if is_test:
            # For test data, we expect a specific directory structure:
            # test_papers/
            #   publishable/
            #   non_publishable/
            labels = []
            filtered_paths = []
            filtered_str_paths = []
            
            for p, sp in zip(paths, str_paths):
                if "publishable" in p.parts:
                    labels.append(1)
                    filtered_paths.append(p)
                    filtered_str_paths.append(sp)
                elif "non_publishable" in p.parts:
                    labels.append(0)
                    filtered_paths.append(p)
                    filtered_str_paths.append(sp)
            
            # Ensure balanced test set
            pos_indices = [i for i, l in enumerate(labels) if l == 1]
            neg_indices = [i for i, l in enumerate(labels) if l == 0]
            
            # Take equal number of positive and negative samples
            sample_size = min(len(pos_indices), len(neg_indices), size // 2)
            
            if sample_size == 0:
                logger.warning(f"Insufficient test data in both classes")
                return [], []
            
            selected_indices = pos_indices[:sample_size] + neg_indices[:sample_size]
            return ([filtered_str_paths[i] for i in selected_indices],
                    [labels[i] for i in selected_indices])
        else:
            # For training data, use the original logic
            if len(str_paths) > size:
                str_paths = str_paths[:size]
                paths = paths[:size]
            
            # Determine true labels based on path
            labels = [1 if PUBLISHABLE_PATH in p.parents else 0 for p in paths]
            
            return str_paths, labels
            
    except Exception as e:
        logger.error(f"Error accessing directory {directory_path}: {str(e)}")
        return [], []

def process_papers(paths: list[str], api_keys: list[str], use_cache: bool = True) -> pw.Table:
    """Process a list of papers and extract features"""
    if not paths:
        logger.warning(f"No paths provided for processing")
        return None
    
    logger.info(f"Processing {len(paths)} papers")
    
    # Create tables for samples
    papers_table = process_directory(paths)
    
    # Extract features using Gemini
    processed_table = extract_features(papers_table, api_keys, use_cache=use_cache)
    
    return processed_table

def train_publishability_classifier(api_keys: list[str], use_cache: bool = True) -> PublishabilityClassifier:
    """Train the publishability classifier on reference data"""
    logger.info(colored("Training publishability classifier...", "cyan"))
    
    # Get sample paths for publishable and non-publishable papers
    publishable_paths, _ = get_sample_paths(PUBLISHABLE_PATH)
    non_publishable_paths, _ = get_sample_paths(NON_PUBLISHABLE_PATH)
    
    if not publishable_paths or not non_publishable_paths:
        raise ValueError("Insufficient training data for publishability classifier")
    
    # Process papers
    publishable_table = process_papers(publishable_paths, api_keys, use_cache)
    non_publishable_table = process_papers(non_publishable_paths, api_keys, use_cache)
    
    if publishable_table is None or non_publishable_table is None:
        raise ValueError("Failed to process papers")
    
    # Run the pathway pipeline to get the data
    pw.run()
    
    # Convert tables to pandas DataFrames
    publishable_df = pw.debug.table_to_pandas(publishable_table)
    non_publishable_df = pw.debug.table_to_pandas(non_publishable_table)
    
    # Combine DataFrames
    all_papers_df = pd.concat([publishable_df, non_publishable_df], ignore_index=True)
    
    # Create labels
    labels = ([1] * len(publishable_paths)) + ([0] * len(non_publishable_paths))
    
    # Initialize and train classifier
    classifier = PublishabilityClassifier()
    classifier.train(all_papers_df, labels)
    
    # Save the trained model
    classifier.save_model()
    
    return classifier

def train_conference_recommender(api_keys: list[str], use_cache: bool = True) -> ConferenceRecommender:
    """Train the conference recommender on publishable papers"""
    logger.info(colored("Training conference recommender...", "cyan"))
    
    # Discover all conference directories
    conference_dirs = {}
    for conf_dir in PUBLISHABLE_PATH.iterdir():
        if conf_dir.is_dir() and list(conf_dir.glob("*.pdf")):
            conference_dirs[conf_dir.name] = conf_dir
    
    if not conference_dirs:
        raise ValueError("No conference directories with PDFs found")
    
    # Process papers from each conference
    conference_papers = {}
    for name, path in conference_dirs.items():
        paths, _ = get_sample_paths(path)
        if paths:
            papers = process_papers(paths, api_keys, use_cache)
            if papers is not None:
                conference_papers[name] = papers
    
    if not conference_papers:
        raise ValueError("No conference papers could be processed for training")
    
    # Initialize and train recommender
    recommender = ConferenceRecommender()
    recommender.train(conference_papers)
    recommender.save_model()
    
    return recommender

def analyze_papers_until_threshold(papers_table: pw.Table, publishability_classifier: PublishabilityClassifier, 
                           conference_recommender: ConferenceRecommender, min_publishable: int = 5) -> list:
    """Analyze papers and continue until we have the minimum required publishable papers"""
    # Get publishability predictions
    pub_predictions = publishability_classifier.predict(papers_table)
    
    # Get conference recommendations for publishable papers
    conf_recommendations = conference_recommender.recommend_conferences(papers_table)
    
    # Convert pathway table to pandas for paper info
    pw.run()
    papers_df = pw.debug.table_to_pandas(papers_table)
    
    results = []
    publishable_count = 0
    
    for idx, (paper_info, pub_pred, conf_recs) in enumerate(
        zip(papers_df.itertuples(), pub_predictions, conf_recommendations)
    ):
        is_publishable = pub_pred['publishable']
        if is_publishable:
            publishable_count += 1
            
        result = {
            'title': f"paper_{idx + 1}",
            'publishable': is_publishable,
            'pub_confidence': pub_pred['confidence'],
            'recommendations': conf_recs if is_publishable else []
        }
        results.append(result)
        
    return results, publishable_count

def display_results(results: list):
    """Display analysis results in a colorful table format"""
    # Prepare table data
    table_data = []
    for result in results:
        title = result['title']
        is_publishable = result['publishable']
        pub_confidence = result['pub_confidence']
        
        # Format publishability status
        pub_status = colored("✓ Publishable", "green") if is_publishable else colored("✗ Not Publishable", "red")
        pub_conf = colored(f"{pub_confidence:.2%}", "cyan")
        
        # Format conference recommendations
        if is_publishable and result.get('recommendations'):
            # Ensure recommendations are properly formatted
            rec_lines = []
            for rec in result['recommendations'][:3]:  # Top 3 recommendations
                try:
                    if isinstance(rec, (list, tuple)) and len(rec) == 2:
                        conf_name, conf_score = rec
                        conf_score_float = float(conf_score)
                        rec_lines.append(f"{conf_name}: {colored(f'{conf_score_float:.2%}', 'yellow')}")
                    elif isinstance(rec, dict):
                        conf_name = rec.get('name', 'Unknown')
                        conf_score = float(rec.get('confidence', 0))
                        rec_lines.append(f"{conf_name}: {colored(f'{conf_score:.2%}', 'yellow')}")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error formatting recommendation: {str(e)}")
                    rec_lines.append(colored("Format Error", "red"))
            rec_text = "\n".join(rec_lines) if rec_lines else colored("No valid recommendations", "red")
        else:
            rec_text = colored("N/A", "red")
        
        table_data.append([title, pub_status, pub_conf, rec_text])
    
    # Print table
    headers = ["Paper", "Status", "Confidence", "Top Conference Recommendations"]
    print("\n" + colored("=== Paper Analysis Results ===", "cyan", attrs=['bold']))
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def load_metrics_history():
    """Load existing metrics history from JSON file"""
    metrics_file = Path("model_metrics.json")
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {'runs': []}

def save_metrics_history(metrics):
    """Save metrics history to JSON file"""
    metrics_file = Path("model_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def calculate_metrics(true_labels, predicted_labels, confidence_scores):
    """Calculate performance metrics with proper handling of edge cases"""
    try:
        # Handle case where there are no predictions
        if not predicted_labels or not true_labels:
            logger.warning("No predictions or true labels available for metrics calculation")
            return {
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

        # Convert inputs to numpy arrays for consistency
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        confidence_scores = np.array(confidence_scores)

        # Calculate metrics with zero_division parameter
        metrics = {
            'f1': float(f1_score(true_labels, predicted_labels, zero_division=0)),
            'precision': float(precision_score(true_labels, predicted_labels, zero_division=0)),
            'recall': float(recall_score(true_labels, predicted_labels, zero_division=0)),
            'accuracy': float(accuracy_score(true_labels, predicted_labels)),
            'avg_confidence': float(np.mean(confidence_scores)) if len(confidence_scores) > 0 else 0.0,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(true_labels),
            'num_positive': int(np.sum(true_labels)),
            'num_predicted_positive': int(np.sum(predicted_labels))
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def display_metrics(metrics):
    """Display metrics in a formatted way"""
    print("\n" + colored("=== Model Performance Metrics ===", "cyan", attrs=['bold']))
    print(f"Number of Samples: {colored(str(metrics.get('num_samples', 'N/A')), 'white')}")
    print(f"True Positive Papers: {colored(str(metrics.get('num_positive', 'N/A')), 'white')}")
    print(f"Predicted Positive Papers: {colored(str(metrics.get('num_predicted_positive', 'N/A')), 'white')}")
    print(f"F1 Score: {colored(f'{metrics['f1']:.3f}', 'yellow')}")
    print(f"Precision: {colored(f'{metrics['precision']:.3f}', 'yellow')}")
    print(f"Recall: {colored(f'{metrics['recall']:.3f}', 'yellow')}")
    print(f"Accuracy: {colored(f'{metrics['accuracy']:.3f}', 'yellow')}")
    print(f"Average Confidence: {colored(f'{metrics['avg_confidence']:.2%}', 'yellow')}")
    if 'error' in metrics:
        print(colored(f"Warning: {metrics['error']}", "red"))

def main():
    parser = argparse.ArgumentParser(description="Train and run paper classifiers")
    parser.add_argument("--use-cache", action="store_true", help="Use cached API responses")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG for enhanced analysis")
    args = parser.parse_args()
    
    try:
        # Get API keys
        api_keys = get_api_keys()
        
        # Get training data
        publishable_paths, publishable_labels = get_sample_paths(TRAIN_PUBLISHABLE_PATH)
        non_publishable_paths, non_publishable_labels = get_sample_paths(TRAIN_NON_PUBLISHABLE_PATH)
        
        # Combine training paths and labels
        train_paths = publishable_paths + non_publishable_paths
        train_labels = publishable_labels + non_publishable_labels
        
        if args.use_rag:
            logger.info(colored("Initializing RAG system...", "cyan"))
            rag_system = ResearchPaperRAG()
            
            # Process training papers for RAG
            publishable_table = process_papers(publishable_paths, api_keys, args.use_cache)
            non_publishable_table = process_papers(non_publishable_paths, api_keys, args.use_cache)
            
            # Run pathway to get data
            pw.run()
            
            # Convert to pandas for RAG indexing
            publishable_df = pw.debug.table_to_pandas(publishable_table)
            non_publishable_df = pw.debug.table_to_pandas(non_publishable_table)
            
            # Prepare papers for RAG
            all_papers = []
            for idx, row in publishable_df.iterrows():
                paper = {
                    "id": f"pub_{idx}",
                    "title": f"Publishable Paper {idx}",
                    "abstract": row.get("abstract", ""),
                    "methodology": row.get("methodology", ""),
                    "results": row.get("results", ""),
                    "conclusion": row.get("conclusion", "")
                }
                all_papers.append(paper)
            
            for idx, row in non_publishable_df.iterrows():
                paper = {
                    "id": f"nonpub_{idx}",
                    "title": f"Non-Publishable Paper {idx}",
                    "abstract": row.get("abstract", ""),
                    "methodology": row.get("methodology", ""),
                    "results": row.get("results", ""),
                    "conclusion": row.get("conclusion", "")
                }
                all_papers.append(paper)
            
            # Index papers in RAG system
            rag_system.index_papers(all_papers)
            logger.info(colored("✓ RAG system initialized", "green"))
        
        # Train models
        if args.use_rag:
            publishability_classifier = rag_system
            conference_recommender = rag_system
        else:
            # Train on combined training data
            train_table = process_papers(train_paths, api_keys, args.use_cache)
            publishability_classifier = train_publishability_classifier(api_keys, args.use_cache)
            conference_recommender = train_conference_recommender(api_keys, args.use_cache)
            logger.info(colored("✓ Models trained", "green"))
        
        # Process test papers with balanced sampling
        test_paths, test_labels = get_sample_paths(PAPERS_PATH, is_test=True)
        if not test_paths:
            logger.error("No test papers found or insufficient balanced test data")
            return
            
        logger.info(f"Processing {len(test_paths)} test papers ({sum(test_labels)} positive, {len(test_labels) - sum(test_labels)} negative)")
        test_table = process_papers(test_paths, api_keys, args.use_cache)
        
        # Get predictions first
        if not args.use_rag:
            pub_predictions = publishability_classifier.predict(test_table)
            # Run pathway to get test data
            pw.run()
            # Convert to pandas
            test_df = pw.debug.table_to_pandas(test_table)
            # Handle predictions that are already lists
            pub_predictions_df = pd.DataFrame(pub_predictions)
        else:
            # Run pathway to get test data
            pw.run()
            test_df = pw.debug.table_to_pandas(test_table)
        
        results = []
        publishable_count = 0
        predicted_labels = []
        confidence_scores = []
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            paper = {
                "id": f"test_{i}",
                "title": f"Test Paper {str(i)}",
                "abstract": str(row.get("abstract", "")),
                "methodology": str(row.get("methodology", "")),
                "results": str(row.get("results", "")),
                "conclusion": str(row.get("conclusion", ""))
            }
            
            if args.use_rag:
                # Use RAG for assessment
                pub_assessment = publishability_classifier.get_publishability_assessment(paper)
                is_publishable = pub_assessment["publishable"]
                pub_confidence = pub_assessment["confidence"]
                
                if is_publishable:
                    publishable_count += 1
                    # Get conference recommendations
                    conf_recs = conference_recommender.get_conference_recommendations(paper)
                    recommendations = [
                        (rec["name"], rec["confidence"])
                        for rec in conf_recs.get("recommended_conferences", [])
                    ]
                else:
                    recommendations = []
            else:
                # Use predictions from DataFrame with integer index
                current_prediction = pub_predictions_df.iloc[i]
                is_publishable = current_prediction["publishable"]
                pub_confidence = current_prediction["confidence"]
                
                if is_publishable:
                    publishable_count += 1
                    # Get conference recommendations for publishable papers
                    paper_table = process_papers([test_paths[i]], api_keys, args.use_cache)
                    conf_recommendations = conference_recommender.recommend_conferences(paper_table)
                    pw.run()
                    
                    # Handle recommendations based on their structure
                    recommendations = []
                    if isinstance(conf_recommendations, list):
                        for rec in conf_recommendations:
                            if isinstance(rec, dict):
                                if 'name' in rec and 'confidence' in rec:
                                    recommendations.append((rec['name'], float(rec['confidence'])))
                            elif isinstance(rec, (list, tuple)) and len(rec) == 2:
                                recommendations.append((str(rec[0]), float(rec[1])))
                else:
                    recommendations = []
            
            # Store prediction and confidence for metrics
            predicted_labels.append(1 if is_publishable else 0)
            confidence_scores.append(float(pub_confidence))
            
            result = {
                "title": f"paper_{str(i)}",
                "publishable": bool(is_publishable),
                "pub_confidence": float(pub_confidence),
                "recommendations": recommendations
            }
            results.append(result)
        
        # Calculate and display metrics
        metrics = calculate_metrics(test_labels, predicted_labels, confidence_scores)
        
        # Display results and metrics
        display_results(results)
        display_metrics(metrics)
        
        # Save metrics history
        metrics_history = load_metrics_history()
        metrics_history['runs'].append(metrics)
        save_metrics_history(metrics_history)
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
