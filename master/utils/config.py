import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
env_path = Path("master/.env")
load_dotenv(dotenv_path=env_path)

# Base paths
BASE_DIR = Path("master")
DATA_DIR = BASE_DIR / "data"

# Data paths
TEST_DATA_DIR = DATA_DIR / "test_papers"
PAPERS_PATH = TEST_DATA_DIR  # Changed to point to test papers
PUBLISHABLE_PATH = DATA_DIR / "Reference/Publishable"
NON_PUBLISHABLE_PATH = DATA_DIR / "Reference/Non-Publishable"

# Training data paths
TRAIN_PUBLISHABLE_PATH = DATA_DIR / "Reference/Publishable"
TRAIN_NON_PUBLISHABLE_PATH = DATA_DIR / "Reference/Non-Publishable"

# API Configuration
def get_api_keys() -> List[str]:
    """Get all available API keys"""
    keys = []
    for i in range(1, 10):  # Support up to 9 keys
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            keys.append(key)
    
    # Add the default key if it exists
    default_key = os.getenv("GOOGLE_API_KEY")
    if default_key:
        keys.append(default_key)
    
    if not keys:
        raise ValueError(
            "No Google API keys found. Please set at least one API key in master/.env file"
        )
    
    return keys 