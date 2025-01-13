"""
Research Paper Classification System
==================================

This package provides functionality for analyzing and classifying research papers
using BERT embeddings and Gemini analysis.
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('paper_processing.log')
    ]
)