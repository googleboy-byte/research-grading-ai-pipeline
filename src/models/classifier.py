import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import logging

def train_classifier(X: pd.DataFrame, y: np.array) -> Pipeline:
    """Train a classifier pipeline with balanced data."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=3)),
        ('classifier', RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Log data shape and class distribution
    logging.info(f"Data shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        logging.info(f"Class {val}: {count} samples")
    
    # Perform cross-validation during training
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0)
    }
    
    cv_results = cross_validate(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    
    # Log cross-validation results during training
    logging.info("\nCross-validation results during training:")
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        logging.info(f"{metric.capitalize()}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    # Fit the final pipeline on all data
    pipeline.fit(X, y)
    return pipeline

def evaluate_classifier(pipeline: Pipeline, X: pd.DataFrame, y: np.array) -> None:
    """Evaluate classifier using cross-validation with multiple metrics."""
    # Perform stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores for multiple metrics
    metrics = {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'f1'
    }
    
    # Collect metrics in a list of dictionaries
    cv_results = []
    for metric_name, scoring in metrics.items():
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        cv_results.append({
            'Metric': metric_name,
            'Mean': f"{cv_scores.mean():.3f}",
            'Std Dev': f"{cv_scores.std():.3f}",
            'Min': f"{cv_scores.min():.3f}",
            'Max': f"{cv_scores.max():.3f}"
        })
    
    # Create and display metrics table
    metrics_df = pd.DataFrame(cv_results)
    print("\nCross-validation Metrics:")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)
    
    # Get predictions for detailed metrics
    y_pred = pipeline.predict(X)
    
    # Print confusion matrix in a more readable format
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print("=" * 40)
    print(f"                  Predicted")
    print(f"                  N    P")
    print(f"Actual  N        {cm[0][0]:<4} {cm[0][1]:<4}")
    print(f"        P        {cm[1][0]:<4} {cm[1][1]:<4}")
    print("=" * 40)
    print(f"N: Non-Publishable, P: Publishable") 