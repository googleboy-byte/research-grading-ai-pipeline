import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pathway as pw
import pandas as pd
import joblib
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def extract_numerical_features(analysis_json: str) -> dict:
    """Extract numerical features from Gemini analysis JSON responses"""
    try:
        # First try to parse the quality_analysis directly
        quality_data = json.loads(analysis_json.strip())
        if isinstance(quality_data, str):
            # If the entire response is a string, try to parse it
            # Clean up the string first
            quality_data = quality_data.replace('\n', ' ').replace('\r', ' ')
            quality_data = json.loads(quality_data)
        
        # Convert string scores to float, with better error handling
        content_quality = quality_data.get("content_quality", "0")
        presentation = quality_data.get("presentation", "0")
        
        # Handle cases where scores might be strings like "8" or "8/10"
        def parse_score(score):
            if isinstance(score, (int, float)):
                return float(score)
            # Remove any non-numeric characters except decimal point
            score = ''.join(c for c in str(score) if c.isdigit() or c == '.')
            return float(score) if score else 0.0
        
        features = {
            "content_quality": parse_score(content_quality),
            "presentation": parse_score(presentation)
        }
        
        logger.debug(f"Extracted numerical features: {features}")
        return features
    except Exception as e:
        logger.warning(f"Error extracting numerical features: {str(e)}")
        return {"content_quality": 0.0, "presentation": 0.0}

def extract_text_features(abstract_json: str, methodology_json: str) -> str:
    """Extract text features for TF-IDF vectorization"""
    text_features = []
    
    def safe_json_loads(json_str: str) -> dict:
        """Safely load JSON string with better error handling"""
        try:
            # Clean up the string first
            json_str = json_str.strip()
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            # Remove any invalid control characters
            json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
            
            data = json.loads(json_str)
            if isinstance(data, str):
                data = json.loads(data)
            return data
        except Exception as e:
            logger.warning(f"Error parsing JSON: {str(e)}")
            return {}
    
    try:
        # Process abstract analysis
        if abstract_json:
            abstract_data = safe_json_loads(abstract_json)
            text_features.extend([
                str(abstract_data.get("main_topic", "")),
                str(abstract_data.get("objective", "")),
                str(abstract_data.get("methodology", "")),
                str(abstract_data.get("key_findings", ""))
            ])
        
        # Process methodology analysis
        if methodology_json:
            method_data = safe_json_loads(methodology_json)
            text_features.extend([
                str(method_data.get("research_type", "")),
                str(method_data.get("methods_used", "")),
                str(method_data.get("data_collection", "")),
                str(method_data.get("analysis_techniques", ""))
            ])
        
        # Combine and clean features
        text = " ".join(filter(None, text_features))
        if not text.strip():
            logger.warning("No text features extracted")
            return "unknown"
        return text
    except Exception as e:
        logger.error(f"Error in text feature extraction: {str(e)}")
        return "unknown"

class ConferenceRecommender:
    def __init__(self, model_dir: str = "master/models/saved"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.tfidf = TfidfVectorizer(
            max_features=100,
            min_df=1,  # Include terms that appear in at least 1 document
            stop_words='english'
        )
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Store conference metadata
        self.conference_labels = {}
        self.conference_metadata = {}
        self.is_fitted = False
        
    def extract_features_from_df(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract features from a DataFrame, with option to fit or transform"""
        # Extract numerical features
        numerical_features = df.apply(
            lambda row: extract_numerical_features(row["quality_analysis"]),
            axis=1
        )
        numerical_features_df = pd.DataFrame(numerical_features.tolist())
        logger.info(f"Extracted numerical features shape: {numerical_features_df.shape}")
        
        # Extract text features
        text_features = df.apply(
            lambda row: extract_text_features(
                row["abstract_analysis"],
                row["methodology_analysis"]
            ),
            axis=1
        )
        logger.info(f"Number of text features extracted: {len(text_features)}")
        
        # Ensure we have some valid text features
        if text_features.str.strip().str.len().sum() == 0:
            logger.error("No valid text features found in any document")
            raise ValueError("No valid text features could be extracted from the documents")
        
        # Transform text features
        if fit:
            text_features_matrix = self.tfidf.fit_transform(text_features)
            numerical_features_scaled = self.scaler.fit_transform(numerical_features_df)
        else:
            text_features_matrix = self.tfidf.transform(text_features)
            numerical_features_scaled = self.scaler.transform(numerical_features_df)
        
        logger.info(f"Text features matrix shape: {text_features_matrix.shape}")
        logger.info(f"Scaled numerical features shape: {numerical_features_scaled.shape}")
        
        # Combine features
        X = np.hstack([
            numerical_features_scaled,
            text_features_matrix.toarray()
        ])
        logger.info(f"Final feature matrix shape: {X.shape}")
        
        return X
        
    def prepare_features(self, table: pw.Table, fit: bool = False) -> np.ndarray:
        """Convert Pathway table to feature matrices"""
        # Run the Pathway pipeline to get the data
        pw.run()
        
        # Convert to pandas for easier processing
        df = pw.debug.table_to_pandas(table)
        logger.info(f"Processing {len(df)} papers")
        
        return self.extract_features_from_df(df, fit=fit)
    
    def train(self, conference_papers: dict[str, pw.Table]):
        """
        Train the classifier on papers from different conferences
        Args:
            conference_papers: Dictionary mapping conference names to their paper tables
        """
        conference_names = []
        
        # First, combine all papers into a single DataFrame for feature extraction
        all_papers_data = []
        for idx, (conf_name, papers) in enumerate(conference_papers.items()):
            logger.info(f"Processing papers for conference: {conf_name}")
            pw.run()  # Ensure the pipeline runs for each table
            df = pw.debug.table_to_pandas(papers)
            df['conference_idx'] = idx
            df['conference_name'] = conf_name
            all_papers_data.append(df)
            
            conference_names.append(conf_name)
            self.conference_labels[idx] = conf_name
        
        # Combine all papers
        all_papers_df = pd.concat(all_papers_data, ignore_index=True)
        
        # Extract features from combined data
        X = self.extract_features_from_df(all_papers_df, fit=True)
        y = all_papers_df['conference_idx'].values
        
        logger.info(f"Training on dataset with shape: {X.shape}")
        
        # Train classifier on all data
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        logger.info("Model training completed")
    
    def recommend_conferences(self, papers_table: pw.Table, top_k: int = 3) -> list:
        """
        Recommend top-k conferences for each paper with confidence scores
        Returns list of (conference_name, confidence) tuples for each paper
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making recommendations")
            
        X = self.prepare_features(papers_table, fit=False)
        probas = self.classifier.predict_proba(X)
        
        recommendations = []
        for paper_probs in probas:
            # Get top-k conferences
            top_indices = np.argsort(paper_probs)[-top_k:][::-1]
            paper_recs = [
                (self.conference_labels[idx], float(paper_probs[idx]))
                for idx in top_indices
            ]
            recommendations.append(paper_recs)
        
        return recommendations
    
    def save_model(self, model_name: str = "conference_recommender"):
        """Save the trained model and its components"""
        if not self.is_fitted:
            raise ValueError("Cannot save an untrained model")
            
        model_path = self.model_dir / f"{model_name}"
        os.makedirs(model_path, exist_ok=True)
        
        # Save components
        joblib.dump(self.tfidf, model_path / "tfidf.joblib")
        joblib.dump(self.scaler, model_path / "scaler.joblib")
        joblib.dump(self.classifier, model_path / "classifier.joblib")
        
        # Save conference metadata
        metadata = {
            "conference_labels": self.conference_labels,
            "conference_metadata": self.conference_metadata
        }
        joblib.dump(metadata, model_path / "metadata.joblib")
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = "conference_recommender"):
        """Load a trained model and its components"""
        model_path = self.model_dir / f"{model_name}"
        
        # Load components
        self.tfidf = joblib.load(model_path / "tfidf.joblib")
        self.scaler = joblib.load(model_path / "scaler.joblib")
        self.classifier = joblib.load(model_path / "classifier.joblib")
        
        # Load conference metadata
        metadata = joblib.load(model_path / "metadata.joblib")
        self.conference_labels = metadata["conference_labels"]
        self.conference_metadata = metadata["conference_metadata"]
        
        self.is_fitted = True
        logger.info(f"Model loaded from {model_path}") 

class PublishabilityClassifier:
    def __init__(self, model_dir: str = "master/models/saved"):
        """Initialize the publishability classifier"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def extract_features_from_df(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract features from DataFrame for publishability classification"""
        # Extract text features
        text_features = []
        for _, row in df.iterrows():
            try:
                # Extract abstract features
                abstract_data = json.loads(row.get('abstract_analysis', '{}'))
                abstract_text = ' '.join([
                    str(abstract_data.get('main_topic', '')),
                    str(abstract_data.get('objective', '')),
                    str(abstract_data.get('methodology', '')),
                    str(abstract_data.get('key_findings', ''))
                ])
                
                # Extract methodology features
                method_data = json.loads(row.get('methodology_analysis', '{}'))
                method_text = ' '.join([
                    str(method_data.get('research_type', '')),
                    str(method_data.get('methods_used', '')),
                    str(method_data.get('data_collection', '')),
                    str(method_data.get('analysis_techniques', ''))
                ])
                
                # Combine features
                text_features.append(f"{abstract_text} {method_text}")
            except Exception as e:
                logger.warning(f"Error extracting text features: {str(e)}")
                text_features.append("")
        
        # Transform text features
        if fit:
            text_features_vectorized = self.tfidf_vectorizer.fit_transform(text_features)
        else:
            text_features_vectorized = self.tfidf_vectorizer.transform(text_features)
        
        # Extract numerical features
        numerical_features = []
        for _, row in df.iterrows():
            try:
                quality_data = json.loads(row.get('quality_analysis', '{}'))
                numerical_features.append({
                    'content_quality': float(quality_data.get('content_quality', 0)),
                    'presentation': float(quality_data.get('presentation', 0))
                })
            except Exception as e:
                logger.warning(f"Error extracting numerical features: {str(e)}")
                numerical_features.append({
                    'content_quality': 0.0,
                    'presentation': 0.0
                })
        
        # Convert to numpy array
        numerical_features_array = np.array([
            [features['content_quality'], features['presentation']]
            for features in numerical_features
        ])
        
        # Scale numerical features
        if fit:
            numerical_features_scaled = self.scaler.fit_transform(numerical_features_array)
        else:
            numerical_features_scaled = self.scaler.transform(numerical_features_array)
        
        # Combine features
        return np.hstack([
            text_features_vectorized.toarray(),
            numerical_features_scaled
        ])
    
    def train(self, df: pd.DataFrame, labels: list):
        """Train the publishability classifier"""
        logger.info("Training publishability classifier...")
        
        # Prepare features
        X = self.extract_features_from_df(df, fit=True)
        y = np.array(labels)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
    
    def predict(self, table: pw.Table) -> list:
        """Predict publishability for papers"""
        # Convert pathway table to pandas DataFrame
        pw.run()
        df = pw.debug.table_to_pandas(table)
        
        # Extract features
        X = self.extract_features_from_df(df, fit=False)
        
        # Make predictions
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                'publishable': bool(pred),
                'confidence': float(prob[1] if pred else prob[0])
            })
        
        return results
    
    def save_model(self, model_name: str = "publishability_classifier"):
        """Save the trained model and its components"""
        model_path = self.model_dir / f"{model_name}.joblib"
        tfidf_path = self.model_dir / f"{model_name}_tfidf.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.tfidf_vectorizer, tfidf_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str = "publishability_classifier"):
        """Load a trained model and its components"""
        model_path = self.model_dir / f"{model_name}.joblib"
        tfidf_path = self.model_dir / f"{model_name}_tfidf.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        
        if not all(p.exists() for p in [model_path, tfidf_path, scaler_path]):
            raise FileNotFoundError("Model files not found. Train the model first.")
        
        self.classifier = joblib.load(model_path)
        self.tfidf_vectorizer = joblib.load(tfidf_path)
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded from {model_path}") 